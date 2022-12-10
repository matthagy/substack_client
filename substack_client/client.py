import gzip
import json
import os.path
import shutil
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Iterable

import pandas as pd
import secretstorage
import sqlalchemy as sa
from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2

from .util import save_using_tmp, random_sleep, expand_comments, get_file_stats

COOKIE_PATH = os.path.expanduser('~/.config/google-chrome/Default/Cookies')
COOKIE_TMP_PATH = '/tmp/subatck_client_cookies'

CHROME_PASSWORD_LABEL = 'Chrome Safe Storage'
CHROME_PASSWORD_ITERATIONS = 1
CHROME_PASSWORD_SALT = b'saltysalt'
CHROME_PASSWORD_KEY_LEN = 16
CHROME_PASSWORD_IV = b' ' * CHROME_PASSWORD_KEY_LEN

HTTP_HEADERS = {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9",
    "accept-encoding": "gzip",
    "cache-control": "no-cache",
    "pragma": "no-cache",
    "sec-ch-ua": "\" Not A;Brand\";v=\"99\", \"Chromium\";v=\"101\", \"Google Chrome\";v=\"101\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Linux\"",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.41 Safari/537.36"
}


def fetch_site_cookies(site: str) -> pd.DataFrame:
    shutil.copy(COOKIE_PATH, COOKIE_TMP_PATH)
    engine = sa.create_engine(f'sqlite:///' + COOKIE_TMP_PATH)
    sql = '''SELECT * FROM cookies WHERE (host_key = ? OR host_key =?);'''
    df = pd.read_sql_query(sql, engine, params=[site, '.' + site])
    for col in pd.Series(df.columns).pipe(lambda cs: cs[cs.str.endswith('_utc')]):
        df[col] = df[col].map(lambda s: datetime(1601, 1, 1) + timedelta(microseconds=s))
    return df


def fetch_chrome_password() -> str:
    bus = secretstorage.dbus_init()
    collection = secretstorage.get_default_collection(bus)
    for item in collection.get_all_items():
        if item.get_label() == CHROME_PASSWORD_LABEL:
            return item.get_secret().decode('utf8')
    else:
        raise Exception('Chrome password not found!')


def decrypt_cookie(encrypted: bytes, chrome_password: str) -> str:
    version = encrypted[:3]
    if version == b'v10':
        chrome_password = 'peanuts'
    elif version != b'v11':
        raise RuntimeError(f'unknown version {version}')
    key = PBKDF2(chrome_password, CHROME_PASSWORD_SALT, CHROME_PASSWORD_KEY_LEN, CHROME_PASSWORD_ITERATIONS)
    cipher = AES.new(key, AES.MODE_CBC, IV=CHROME_PASSWORD_IV)
    decrypted = cipher.decrypt(encrypted[3:])
    return decrypted[:-decrypted[-1]].decode('utf8')


def build_site_cookie(site: str) -> str:
    df = fetch_site_cookies(site)
    assert len(df), 'empty cookies'
    assert {'connect.sid', 'substack.sid'} & set(df['name']), 'no sid cookie'
    chrome_password = fetch_chrome_password()
    values = df['encrypted_value'].map(lambda encrypted: decrypt_cookie(encrypted, chrome_password))
    return '; '.join(f'{name}={value}' for name, value in zip(df['name'], values))


class SubstackClient:
    hostname: str
    cookie: str

    def __init__(self, hostname: str, cookie: str):
        self.hostname = hostname
        self.cookie = cookie

    @classmethod
    def create(cls, hostname: str, *, cookie_hostname: Optional[str] = None) -> 'SubstackClient':
        return cls(hostname, build_site_cookie(hostname if cookie_hostname is None else cookie_hostname))

    def get_http(self, path: str, *, referer: str = '/') -> str:
        http_headers = HTTP_HEADERS.copy()
        http_headers["cookie"] = self.cookie
        http_headers["Referer"] = self.make_url(referer)
        request = urllib.request.Request(self.make_url(path), method='GET', headers=http_headers)
        with urllib.request.urlopen(request) as response:
            body: bytes = response.read()
        encoding: str = response.headers['Content-Encoding']
        if encoding == 'gzip':
            body = gzip.decompress(body)
        elif encoding:
            raise RuntimeError(f'unknown encoding {encoding!r}')
        return body.decode('utf8')

    def get_json(self, path: str, *, referer: str = '/') -> Any:
        return json.loads(self.get_http(path=path, referer=referer))

    def make_url(self, path: str) -> str:
        if path.startswith('http'):
            return path
        assert path.startswith('/'), repr(path)
        return f'https://{self.hostname}{path}'

    def get_new_posts(self, offset: int, *, limit: int = 12) -> list[dict[str, Any]]:
        assert offset >= 0, offset
        assert limit > 0, limit
        posts = self.get_json(path=f'/api/v1/archive?sort=new&search=&offset={offset}&limit={limit}')
        assert isinstance(posts, list), type(posts)
        return posts

    def get_comments(self, post_id: int, *, referer: str = '/') -> list[dict[str, Any]]:
        result = self.get_json(f'/api/v1/post/{post_id}/comments?token=&all_comments=true&sort=best_first',
                               referer=referer)
        assert isinstance(result, dict)
        comments = result.get('comments', [])
        assert isinstance(comments, list)
        return comments


def fetch_new_posts(client: SubstackClient, full: bool, min_overlap: int, posts_path: Path, limit: int = 12):
    assert min_overlap > 0, min_overlap

    df: Optional[pd.DataFrame] = None
    if posts_path.exists():
        print('loading existing path', posts_path)
        df = pd.read_pickle(posts_path)
    else:
        print(f'no existing posts; path {posts_path} does not exist')

    offset = 0
    overlap_count = 0

    while True:
        new_posts = client.get_new_posts(offset=offset, limit=limit)
        if not new_posts:
            print('no new posts')
            break

        append_df = pd.DataFrame(new_posts).set_index('id')
        append_df['post_date'] = pd.to_datetime(append_df['post_date'])
        overlap = set(append_df.index) & set(df.index) if df is not None else set()
        print(f'fetched {len(append_df)} posts. overlap={len(overlap)}')
        print(append_df[['post_date', 'type', 'comment_count', 'reactions', 'title']]
              .assign(post_date=lambda x: x['post_date'].dt.round('S').dt.tz_localize(None))
              .to_string(line_width=280, max_colwidth=60, justify='left'))

        if df is None:
            df = append_df
        else:
            if overlap:
                df = df.drop(list(overlap), axis=0)
            df = pd.concat([df, append_df]).sort_index()
        save_using_tmp(posts_path, df.to_pickle)

        overlap_count += len(overlap)
        if (not full) and overlap_count >= min_overlap:
            print(f'found {overlap_count=}; exiting early')
            break

        offset += limit

        random_sleep(1.5, 7.5)


def fetch_post_contents(client: SubstackClient, posts: pd.DataFrame, posts_dir: Path, hostname: str):
    posts_dir.mkdir(exist_ok=True)

    def post_path(post_id: int) -> Path:
        return posts_dir / f'{post_id}.html'

    posts = posts.sort_values('post_date', ascending=False)
    need_fetched_post_id = [post_id for post_id in posts.index
                            if not post_path(post_id).exists()]
    print(f'need to fetch {len(need_fetched_post_id)} out of {len(posts)} posts')

    for post_id in need_fetched_post_id:
        post: pd.Series = posts.loc[post_id]
        print(f'post={post_id} on {post["post_date"]}: {post["title"][:30:]}')

        content = client.get_http(post['canonical_url'], referer=f'https://{hostname}/archive?sort=new')
        print(f'fetched {len(content):,} bytes')

        def save_func(path: Path):
            with path.open(mode='wt') as fp:
                fp.write(content)

        save_using_tmp(post_path(post_id), save_func)

        random_sleep(1.5, 7.5)


class CommentFetcher:
    comments_dir: Path
    posts: pd.DataFrame
    client: SubstackClient

    def __init__(self, comments_dir: Path, posts: pd.DataFrame, client: SubstackClient):
        self.comments_dir = comments_dir
        self.posts = posts
        self.client = client

    def comments_path(self, post_id: int) -> Path:
        return self.comments_dir / f'{post_id}.p'

    def get_all_comment_paths(self) -> Iterable[Path]:
        return self.comments_dir.glob('*.p')

    def get_comment_file_mtimes(self) -> pd.DataFrame:
        df = get_file_stats(map(str, self.get_all_comment_paths()))
        df['post_id'] = df['path'].map(lambda path: int(os.path.basename(path)[:-2]))
        return pd.merge(
            self.posts,
            df.set_index('post_id')[['mtime']],
            left_index=True,
            right_index=True
        )

    def fetch_and_save_posts(self, post_ids: Iterable[int]):
        self.comments_dir.mkdir(exist_ok=True)
        for post_id in post_ids:
            self.fetch_and_save_post(post_id)
            random_sleep(minimum=7, maximum=18)

    def fetch_and_save_post(self, post_id: int):
        post: pd.Series = self.posts.loc[post_id]
        print(f'post={post_id} on {post["post_date"]}: {post["title"][:80:]}')
        comments = self.client.get_comments(post_id, referer=post['canonical_url'])
        df = expand_comments(comments)
        print(f'fetched {len(comments)} top-level comments and {len(df)} expanded')
        save_using_tmp(self.comments_path(post_id), df.to_pickle)

    def fetch_missing(self):
        missing_post_ids = [post_id for post_id in self.posts.index
                            if not self.comments_path(post_id).exists()]
        print(f'need to fetch {len(missing_post_ids)} out of {len(self.posts)} posts')
        self.fetch_and_save_posts(missing_post_ids)

    def fetch_stale(self, stale_comment_age: pd.Timedelta):
        post_mtimes = self.get_comment_file_mtimes()
        fetch_age = post_mtimes['mtime'] - post_mtimes['post_date']
        stale_post_ids = fetch_age[fetch_age < stale_comment_age].index
        print(f'fetching {len(stale_post_ids)} stale posts out of {len(self.posts)} posts')
        posts: pd.DataFrame = self.posts.loc[stale_post_ids][['post_date', 'title']]
        print(posts.to_string(index=True, line_width=160, max_colwidth=60))
        self.fetch_and_save_posts(stale_post_ids)

    def fetch_oldest(self, max_post_date: pd.Timestamp):
        post_mtimes = self.get_comment_file_mtimes()
        post_mtimes = post_mtimes[post_mtimes['type'] == 'newsletter']
        post_mtimes = post_mtimes[post_mtimes['reaction']]
        post_mtimes = post_mtimes[post_mtimes['post_date'] < max_post_date]
        post_ids = post_mtimes['mtime'].sort_values(ascending=True).index
        print(f'{len(post_ids)=}')
        self.fetch_and_save_posts(post_ids)
