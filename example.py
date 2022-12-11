from time import sleep

from bs4 import BeautifulSoup

from substack_client import SubstackClient
from substack_client.util import extract_contents

# Create a client for fetching contents from Slow Boring using Chrome browser cookies for authentication
client = SubstackClient.create('www.slowboring.com')

# Fetch metadata for the newest posts
posts = client.get_new_posts(offset=0)
print(f'fetched metadata for {len(posts)} posts')

# Find the newest free newsletter (i.e., not thread) post
for post in posts:
    if post['type'] == 'newsletter' and post['audience'] == 'everyone':
        break
else:
    raise RuntimeError('no free newsletter posts')

print(f'first post: id={post["id"]} title={post["title"]!r} date={post["post_date"]}')

# Fetch the HTML contents of the first post
html_contents = client.get_http(post['canonical_url'])
print(f'fetched {len(html_contents):,} bytes')

# Extract the post text
contents_element = extract_contents(BeautifulSoup(html_contents, features="html.parser"))
text_contents = contents_element.get_text(separator=' ', strip=True)
print(f'post text preview: {text_contents[:100]!r}')

# Fetch the comments for this post
sleep(1)
comments = client.get_comments(post["id"])
print(f'fetch {len(comments)} comments')

# Find my first comment
for comment in comments:
    if comment['name'] == 'Matt Hagy':
        break
else:
    raise RuntimeError('no comments from Matt Hagy')
print(f'comment: id={comment["id"]} date={comment["date"]} reactions={comment["reactions"]}')
print(f'comment preview: {comment["body"][:100]!r}')
