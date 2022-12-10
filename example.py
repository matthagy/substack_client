from pprint import pprint
from time import sleep
from substack_client import SubstackClient

client = SubstackClient.create('www.slowboring.com')
posts = client.get_new_posts(offset=0)
pprint(posts)
sleep(1)
comments = client.get_comments(posts[0]['id'])
pprint(comments)


