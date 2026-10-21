import asyncio,uuid
import httpx
from machina.core.provider.direct import DirectProvider
from machina.core.provider.specs import ProviderSpec
from machina.core.usage import usage_store
from machina.core.provider.caches import _cache_clear
async def main():
 reservations=[]; finishes=[]
 async def reserve(*a,**kw):
  reservations.append(kw.get('purpose','agent'));return uuid.uuid4()
 async def finish(*a,**kw):finishes.append(kw);return True
 usage_store.reserve=reserve;usage_store.finish=finish
 p=DirectProvider(ProviderSpec('fake','https://example.invalid','FAKE_API_KEY','fake'),api_key='fake')
 await p.client.aclose()
 posts=[]
 def handle(req):
  posts.append(req);return httpx.Response(200,json={'model':'fake','choices':[{'message':{'content':'OK'}}],'usage':{'prompt_tokens':10,'completion_tokens':1,'total_tokens':11}})
 p.client=httpx.AsyncClient(base_url='https://example.invalid',transport=httpx.MockTransport(handle))
 class Limit:
  async def acquire(self):pass
 p._rate_limiter=Limit();_cache_clear()
 await usage_store.complete_call(p,uuid.uuid4(),'harness',[{'role':'user','content':'hi'}],tools=[])
 print({'network_posts':len(posts),'reservations':len(reservations),'finishes':len(finishes),'purposes':reservations})
 await p.close()
asyncio.run(main())
