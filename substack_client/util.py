import os
import os.path
import random
import time
from collections import Counter
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Any, Iterable, TypeVar, Optional, Sized, Union, TextIO

import pandas as pd
from bs4 import BeautifulSoup, PageElement
from tqdm import tqdm

T = TypeVar('T')
R = TypeVar('R')
COMMENT_TYPE = dict[str, Any]

POOL: Optional[Pool] = None


def save_using_tmp(filename: Union[str, Path], save_func: Callable[[Union[str, Path]], Any]):
    if not isinstance(filename, Path):
        filename = Path(filename)
    tmp_path = filename.parent / (filename.name + '-tmp')
    save_func(tmp_path)
    tmp_path.rename(filename)


def only(xs: Iterable[T]) -> T:
    it = iter(xs)
    try:
        x = next(it)
    except StopIteration:
        raise ValueError('empty')
    try:
        x2 = next(it)
    except StopIteration:
        return x
    else:
        n = len_optional(xs)
        raise ValueError(f'multiple values {x=} {x2=} len={n if n is not None else "?"}')


def len_optional(xs: Union[Iterable, Sized]) -> Optional[int]:
    try:
        return len(xs)
    except TypeError:
        return None


def random_sleep(minimum=15, maximum=40):
    t = max(minimum, random.uniform(minimum, maximum) + random.gauss(mu=0, sigma=1))
    print(f'sleeping {t:0.1f}')
    time.sleep(t)


def iter_with_progress(xs: Iterable[T], *, desc: Optional[str] = None, total: Optional[int] = None,
                       show_items: bool = False) -> Iterable[T]:
    if total is None:
        total = len_optional(xs)
    with tqdm(iterable=xs, desc=desc, total=total) as t:
        if not show_items:
            yield from t
        else:
            prefix = '' if desc is None else ' ' + desc
            for x in t:
                t.set_description(prefix + str(x))
                yield x


def get_pool() -> Pool:
    global POOL
    if POOL is None:
        POOL = Pool()
    return POOL


def pool_mapper(x: tuple[int, T], *, func: Callable[[T], R]) -> tuple[int, R]:
    return x[0], func(x[1])


def pool_map(func: Callable[[T], R], xs: Iterable[T], *,
             desc: Optional[str] = None, chunksize: int = 1, show_items: bool = False) -> list[R]:
    results = get_pool().imap_unordered(partial(pool_mapper, func=func), enumerate(xs), chunksize=chunksize)
    results_dict = dict(iter_with_progress(results, desc=desc, total=len_optional(xs), show_items=show_items))
    return [results_dict[i] for i in range(len(results_dict))]


def get_file_stats(filepaths: Iterable[str]) -> pd.DataFrame:
    def stat_dict(path: str) -> dict[str, Any]:
        st = os.stat(path)
        d = {'path': path}
        for n in dir(st):
            if n.startswith('st_') and not n.endswith('_ns'):
                d[n[3::]] = getattr(st, n)
        return d

    df = pd.DataFrame([stat_dict(p) for p in filepaths])
    if len(df):
        for col in ['atime', 'mtime', 'ctime']:
            df[col] = pd.to_datetime(df[col], unit='s', utc=True)
    return df


def generate_comments(comment: COMMENT_TYPE) -> Iterable[COMMENT_TYPE]:
    cp = comment.copy()
    children = cp.get('children', [])
    cp['children'] = len(children)
    yield cp
    for child in children:
        yield from generate_comments(child)


def expand_comments(comments: Iterable[COMMENT_TYPE]) -> pd.DataFrame:
    df = pd.DataFrame([ec for c in comments for ec in generate_comments(c)])
    if len(df):
        df = df.drop(['post_id', 'publication_id', 'type'], axis=1)
        df['user_id'] = df['user_id'].fillna(-1).astype(int)
        df['id'] = df['id'].fillna(-1).astype(int)
        for col in 'date', 'edited_at':
            df[col] = pd.to_datetime(df[col])
        for col in 'reactor_names', 'bans':
            df[col] = df[col].map(tuple)
    return df


def load_comments(path: Union[Path, str]) -> pd.DataFrame:
    comments: pd.DataFrame = pd.read_pickle(path)
    if len(comments):
        comments['post_id'] = int(os.path.basename(path).replace('.p', ''))
        comments['id'] = comments['id'].fillna(-1).astype(int)
    return comments


def load_all_comments(paths: Iterable[Union[Path, str]]) -> pd.DataFrame:
    paths = list(paths)
    print('collecting')
    results = pool_map(load_comments, paths, chunksize=max(1, len(paths) // 128))
    print('concat')
    df: pd.DataFrame = pd.concat(list(results), axis=0, ignore_index=True)
    if len(df):
        print('transform')
        df['edited_at'] = pd.to_datetime(df['edited_at'])
        df['likes'] = df['reactions'].map(lambda rs: rs.get('❤', 0))
        df['ancestor_path'] = df['ancestor_path'].map(lambda x: tuple(map(int, x.split('.'))) if x else ())
        df['parent_id'] = df['ancestor_path'].map(lambda x: x[-1] if x else None).astype(pd.Int64Dtype())
        df['thread_id'] = df.apply(lambda x: x['ancestor_path'][0] if x['ancestor_path'] else x['id'],
                                   axis=1).astype(int)
        df = df[['post_id'] + [c for c in df.columns if c != 'post_id']]
    return df


def load_article_content(post_id: int, posts_dir: Path) -> PageElement:
    with posts_dir.joinpath(f'{post_id}.html').open('rt') as fp:
        soup = BeautifulSoup(fp, features="html.parser")
    found = soup.find_all(name='div', attrs={'class': 'available-content'})
    if len(found) != 1:
        raise ValueError(f'{post_id=} had {len(found)=} content elements')
    content, = found
    return content


def extract_text(post_id: int, *, posts_dir: Path) -> tuple[int, str]:
    content = load_article_content(post_id, posts_dir)
    text = content.get_text(separator=' ', strip=True)
    return post_id, text


def extract_posts_text(post_ids: Iterable[int], posts_dir: Path) -> pd.Series:
    extracted = pool_map(partial(extract_text, posts_dir=posts_dir), post_ids)
    return pd.Series(dict(extracted))


def extract_links(post_id: int, *, posts_dir: str) -> pd.DataFrame:
    content = load_article_content(post_id, posts_dir)
    return pd.DataFrame(
        [[post_id, link.attrs['href'], link.get_text(separator=' ', strip=True)] for link in content.find_all('a')],
        columns=['post_id', 'link', 'text']
    )


def extract_posts_links(post_ids: Iterable[int], posts_dir: str) -> pd.DataFrame:
    extracted = pool_map(partial(extract_links, posts_dir=posts_dir), post_ids)
    return pd.concat(extracted, axis=0, ignore_index=True)


def select_last_name(df: pd.DataFrame) -> str:
    return df['name'].loc[df['date'].idxmax()]


def normalize_names(comments_df: pd.DataFrame) -> pd.Series:
    comments_df = comments_df[~(comments_df['name'].isna() |
                                comments_df['body'].isna())]
    user_names = comments_df.groupby('user_id').apply(select_last_name).rename('name')
    name_user_count = user_names.reset_index().groupby('name')['user_id'].nunique().rename('count')
    name_collisions = name_user_count[name_user_count > 1].to_frame().reset_index()['name']
    collisions_mask = user_names.isin(name_collisions)
    user_names[collisions_mask] = [f'{name}:{user_id}' for user_id, name in user_names[collisions_mask].items()]
    return user_names


class TagManager:
    path: Path
    tags: dict[int, set[str]]
    append_io: TextIO

    @classmethod
    def create(cls, path: Path) -> 'TagManager':
        tag_manager = cls(path)
        tag_manager.load()
        return tag_manager

    def __init__(self, path: Path):
        self.path = path
        self.tags = {}

    def load(self):
        if self.path.exists():
            with open(self.path, 'rt') as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    action, comment_id, tag, _ = line.split()
                    comment_id = int(comment_id)
                    if action == 'add':
                        self._add(comment_id, tag)
                    else:
                        assert action == 'del'
                        self._del(comment_id, tag)
        self.append_io = open(self.path, 'at')

    def get_tags(self, comment_id: int) -> frozenset[str]:
        try:
            return frozenset(self.tags[comment_id])
        except KeyError:
            return frozenset()

    def add_tag(self, comment_id: int, tag: str):
        if self._add(comment_id, tag):
            self._record('add', comment_id, tag)

    def del_tag(self, comment_id: int, tag: str):
        if self._del(comment_id, tag):
            self._record('del', comment_id, tag)

    def get_all_tags(self) -> dict[str, int]:
        return Counter(tag
                       for comment_tags in self.tags.values()
                       for tag in comment_tags)

    def get_comments_tags(self, comment_ids: Iterable[int]) -> dict[str, int]:
        rel_comment_ids = set(comment_ids) & self.tags.keys()
        return Counter(tag
                       for comment_id in rel_comment_ids
                       for tag in self.tags[comment_id])

    def filter_for_tags(self, comment_ids: Iterable[int], tags: Iterable[str]) -> set[int]:
        rel_comment_ids = set(comment_ids) & self.tags.keys()
        tags_set = frozenset(tags)
        return {comment_id for comment_id in rel_comment_ids
                if not self.tags[comment_id].isdisjoint(tags_set)}

    def get_tagged_comments(self) -> frozenset[int]:
        return frozenset(self.tags)

    def get_tags_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_records([
            (comment_id, tag)
            for comment_id, comment_tags in self.tags.items()
            for tag in comment_tags
        ], columns=['comment_id', 'tag'])

    def _record(self, action: str, comment_id: int, tag: str):
        print(action, comment_id, tag, round(time.time(), 3),
              file=self.append_io, flush=True)

    def _add(self, comment_id: int, tag: str) -> bool:
        tag = self._norm_tag(tag)
        try:
            comment_tags = self.tags[comment_id]
        except KeyError:
            comment_tags = self.tags[comment_id] = set()
        if tag in comment_tags:
            return False
        else:
            comment_tags.add(tag)
            return True

    def _del(self, comment_id: int, tag: str) -> bool:
        tag = self._norm_tag(tag)
        try:
            comment_tags = self.tags[comment_id]
            comment_tags.remove(tag)
        except KeyError:
            return False
        else:
            if not comment_tags:
                del self.tags[comment_id]
            return True

    @staticmethod
    def _norm_tag(tag: str):
        assert tag
        tag = tag.lower()
        return tag
