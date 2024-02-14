from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import argparse

app = FastAPI(title="builder-runner", version="0.1.0", description="API to run code from the SPAI builder and provide outputs.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TODO
# esta api recibirá desde la UI el grafo 
# esta api envia el grafo a la api de SPAI y recibe código a ejecutar
# esta api ejecuta el código (guardando los outputs en el storage asoicado a el proyecto del usuario)
# esta api expone lo necesario para que la UI del builder pueda obtener los outputs de la ejecución (layers, analytics, etc)

@app.get("/")
async def hello():
    return "Hello World!"


# need this to run in background
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
