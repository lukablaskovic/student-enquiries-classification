from topic_classifier import classifier
from qa_system import getPredefinedAnswer
from semantic_search import search
from aiohttp import web
import asyncio
import aiohttp_cors

routes = web.RouteTableDef()

@routes.get("/")
async def get_handler(request):
    return web.json_response({"status": "OK"}, status=200)


@routes.post("/classificator")
async def classify(request):
    try:
        inquiry = await request.json()
        response = classifier(inquiry["text"])

        return web.json_response(response, status=200)
    except Exception as e:
        return web.json_response({"status": "classificator failed", "message": str(e)}, status=500)
    
@routes.post("/answer")
async def getAnswer(request):
    try:
        inquiry = await request.json()
        print(inquiry["text"])
        response = search(inquiry["text"])

        return web.json_response(response, status=200)
    except Exception as e:
        return web.json_response({"status": "get-answer failed", "message": str(e)}, status=500)
    
@routes.post("/predefined-answer")
async def getAnswer2(request):
    try:
        inquiry = await request.json()
        response2 = await getPredefinedAnswer(inquiry["text"])
        
        return web.json_response(response2, status=200)
    except Exception as e:
        return web.json_response({"status": "get-predefined-answer failed", "message": str(e)}, status=500)



app = web.Application()
app.router.add_routes(routes)

# Cors policy set up
cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_headers="*"
    )
})
for route in list(app.router.routes()):
    cors.add(route)


if __name__ == '__main__':
    web.run_app(app, port=8081)
