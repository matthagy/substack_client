# substack_client

A prototype client library for accessing Substack content using the REST API that serves the frontend.

This library is not associated with Substack the company in any way.
Further, it does not use an official API but instead uses a non-documented REST API
that serves the website.
Therefore, the library could break at anytime should that internal API be revised.

You can learn more in the blog post,
[Developing a Substack client to fetch posts and comments](https://matthagy.substack.com/p/developing-a-custom-substack-front).

## Installation
```shell
python setup.py
```

## Example

The `example.py` file demonstrates use of the library.
```python
from pprint import pprint
from time import sleep
from substack_client import SubstackClient

client = SubstackClient.create('www.slowboring.com')
posts = client.get_new_posts(offset=0)
pprint(posts)
sleep(1)
comments = client.get_comments(posts[0]['id'])
pprint(comments)
```