from aiohttp import web
import aiohttp
import asyncio
import json

routes = web.RouteTableDef()


@routes.get("/")
async def get_handler(request):
    return web.json_response({"status" : "OK"}, status=200)

@routes.post("/classificator")
async def classify(request):
    try:
        inquiry = await request.json()
        print(inquiry)
        return web.json_response({"status" : "OK"}, status=200)
    except Exception as e:
        return web.json_response({"status" : "failed", "message" : str(e)}, status=500)
    
    
app = web.Application()

app.router.add_routes(routes)

if __name__ == '__main__':
    web.run_app(app)