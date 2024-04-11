from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import argparse
from spai.storage import Storage
from pydantic import BaseModel, field_validator
import json
from typing import List, Dict
import os
from spai.config import SPAIVars
from spai.data.satellite import explore_satellite_imagery, download_satellite_imagery
from spai.image.xyz import get_image_data, get_tile_data, ready_image
from starlette.responses import StreamingResponse
from spai.image.xyz.errors import ImageOutOfBounds
from typing import List
from dask import get

app = FastAPI(title="builder-runner", version="0.1.0", description="API to run code from the SPAI builder and provide outputs.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

storage = Storage()["data"]
vars = SPAIVars()

os.environ["SH_CLIENT_ID"] = os.getenv("SH_CLIENT_ID", vars["SH_CLIENT_ID"])
os.environ["SH_CLIENT_SECRET"] = os.getenv("SH_CLIENT_SECRET", vars["SH_CLIENT_SECRET"])

@app.get("/")
async def hello():
    print("Builder")
    return storage.list()

# manage aois in storage

class VectorModel(BaseModel):
    type: str  # must be 'Feature'
    properties: Dict  # shall be JSON Object or null
    geometry: Dict  # shall be Geometry Object or null: 'type' member must be a Geometry Type and 'coordinates' member shall be an array

    @field_validator('type')
    def check_feature_type_is_valid(cls, type):
        assert type == 'Feature', 'Invalid GeoJSON: In Feature Object, "type" member is not valid'
        return type

    @field_validator('properties')
    # throws a JSONDecodeError
    def check_feature_properties_is_valid(cls, properties):
        p = json.dumps(properties)
        assert p, 'Invalid GeoJSON: In Feature Object, "properties" member is not valid'
        return properties

    @field_validator('geometry')
    # throws a JSONDecodeError
    # returns error message: value is not a valid dict if geometry is not a JSON file
    def check_feature_geometry_is_valid(cls, geometry):
        geo = json.dumps(geometry)
        geo_members = json.loads(geo)
        assert geo_members['type'] in ('Point', 'MultiPoint', 'LineString', 'MultiLineString', 'Polygon', 'MultiPolygon',
                                       'GeometryCollection'), 'Invalid GeoJSON: In Feature Object, "geometry - type" member is not valid'
        assert isinstance(
            geo_members['coordinates'], list) == True, 'Invalid GeoJSON: In Feature Object, "geometry - coordinates" member is not valid'
        return geometry


class GeoJSONModel(BaseModel):
    type: str  # must be a GeoJSON Type
    features: List[VectorModel]  # shall be a JSON array or an empty array

    @field_validator('type')
    def check_geojson_type_is_valid(cls, type):
        assert type in ('FeatureCollection', 'Feature', 'Point', 'LineString', 'MultiPoint', 'Polygon',
                        'MultiLineString', 'MultiPolygon', 'GeometryCollection'), 'Invalid GeoJSON "type" member'
        return type

class InvalidGeoJSONError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)

class AOIGeoJSONFeatureModel(VectorModel):
    @field_validator('geometry')
    def check_feature_geometry_is_valid(cls, geometry):
        geo = json.dumps(geometry)
        geo_members = json.loads(geo)
        if geo_members['type'] not in ('Polygon', 'MultiPolygon'):
            raise InvalidGeoJSONError(
                f'Invalid GeoJSON: In Feature Object, "geometry - type" member is not valid: Must be a "Polygon" or a "MultiPolygon"')
        if not isinstance(geo_members['coordinates'], list):
            raise InvalidGeoJSONError(
                f'Invalid GeoJSON: In Feature Object, "geometry - coordinates" member is not valid')
        # check all coordinates are valid (lat, lng)
        if geo_members['type'] == "Polygon":
            for coords in geo_members['coordinates'][0]:
                assert len(
                    coords) == 2, f'Invalid GeoJSON: In Feature Object, "geometry - coordinates" member is not valid'
                assert coords[0] >= - \
                    180 and coords[0] <= 180, f'latitude should be between -180 and 180'
                assert coords[1] >= - \
                    90 and coords[1] <= 90, f'longitude should be between -90 and 90'
        elif geo_members['type'] == "MultiPolygon":
            for _coords in geo_members['coordinates'][0]:
                for coords in _coords:
                    assert len(
                        coords) == 2, f'Invalid GeoJSON: In Feature Object, "geometry - coordinates" member is not valid'
                    assert coords[0] >= - \
                        180 and coords[0] <= 180, f'latitude should be between -180 and 180'
                    assert coords[1] >= - \
                        90 and coords[1] <= 90, f'longitude should be between -90 and 90'
        return geometry

class AOIGeoJSONModel(GeoJSONModel):
    type: str
    features: List[AOIGeoJSONFeatureModel]

    @field_validator('features')
    def check_geojson_features_is_valid(cls, features):
        if len(features) != 1:
            raise InvalidGeoJSONError(
                f'Invalid GeoJSON: Features must be a list with 1 element, you provided {len(features)}.')
        return features
    
class AOICreateModel(BaseModel):
    name: str
    geojson: AOIGeoJSONModel

@app.post("/aois")
def create_new_aoi(aoi: AOICreateModel):
    try:
        storage.create(aoi.geojson.model_dump(), f"{aoi.name}.geojson")
    except Exception as e:
        print('ERROR', repr(e))
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=repr(e))

@app.get("/aois")
def retrieve_aois():
    try:
        aois = storage.list("*.geojson")
        return [
            {
                "name": aoi.split(".geojson")[0],
                "features": storage.read(aoi).__geo_interface__ # if geojson, storage returns GeoDataframe
            } for aoi in aois
        ]
    except Exception as e:
        print('ERROR', repr(e))
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=repr(e))

# explore and download images 

class ExploreImages(BaseModel):
    aoi: str
    startDate: str
    endDate: str
    sensor: str
    cloud_coverage: float

@app.post("/images")
def explore_satellite_images(body: ExploreImages):
    try:
        aoi = storage.read(f"{body.aoi}.geojson")
        images = explore_satellite_imagery(
            aoi, 
            date = (body.startDate, body.endDate),
            collection=body.sensor,
            cloud_cover=body.cloud_coverage
        )
        if len(images) == 0:
            raise ValueError("No images found")
        return images
    except Exception as e:
        print('ERROR', repr(e))
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=repr(e))

class DownloadImages(BaseModel):
    aoi: str
    dates: List[str]
    collection: str

@app.post("/images/download")
def download_satellite_images(body: DownloadImages):
    try:
        aoi = storage.read(f"{body.aoi}.geojson")
        for date in body.dates:
            download_satellite_imagery(
                storage,
                aoi, 
                date,
                collection=body.collection,
                name=f"{body.aoi}_{body.collection}_{date}.tif"
            )
        return storage.list("*.tif")
    except Exception as e:
        print('ERROR', repr(e))
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=repr(e))

@app.get("/images")
def retrieve_images():
    try:
        return storage.list("*.tif")
    except Exception as e:
        print('ERROR', repr(e))
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=repr(e))

# visualize images 


@app.get("/{image}/{z}/{x}/{y}.png")
def retrieve_image_tile(
    image: str,
    z: int,
    x: int,
    y: int,
    bands: str = "1",
    stretch: str = "0,1",
    palette: str = "viridis",
):
    image_path = storage.get_path(image)
    tile_size = (256, 256)
    if len(bands) == 1:
        bands = int(bands)
    else:
        bands = tuple([int(band) for band in bands.split(",")])
    stretch = tuple([float(v) for v in stretch.split(",")])
    try:
        tile = get_tile_data(image_path, (x, y, z), bands, tile_size)
        tile = get_image_data(tile, stretch, palette)
        image = ready_image(tile)
        return StreamingResponse(image, media_type="image/png")
    except ImageOutOfBounds as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=error.message)
    
# builder 

class GraphBody(BaseModel):
    nodes: List[dict]
    edges: List[dict]

@app.post("/graph")
def save_graph(body: GraphBody):
    try:
        # Storage de momento no guarda listas... pero es un json...
        storage.create_from_dict(body.nodes, "nodes.json")
        storage.create_from_dict(body.edges, "edges.json")
        return 
    except Exception as e:
        print('ERROR', repr(e))
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=repr(e))

@app.get("/graph")
def retrieve_graph():
    try:
        nodes = storage.read("nodes.json").fillna(0).to_dict(orient="records")
        edges = storage.read("edges.json").fillna(0).to_dict(orient="records")
        # al leer el json con pandas, los campos vacios se rellenan con nan (los cambio a 0)
        # esto puede afectar si alguna propiedad no espera un valor como 0...
        # el storage debería  exponer una manera de leer un json directamente, sin pasar por pandas !!!
        print(nodes)
        return {
            "nodes": nodes,
            "edges": edges
        }
    except Exception as e:
        print('ERROR', repr(e))
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=repr(e))

@app.post("/graph/run")
def save_graph(body: GraphBody):
    try:
        # save graph
        nodes = storage.read("nodes.json").to_dict(orient="records")
        edges = storage.read("edges.json").to_dict(orient="records")
        # convert to dask graph
        graph = {}
        for node in body.nodes:
            data = node['data']
            fn = nodeId2Function(data['id'])
            args = [field['value'] for field in data['fields']]
            # find inputs in edges 
            inputs = data['inputs']
            _edges = [edge for edge in edges if edge['source'] == node['id']]
            for input in inputs:
                input_name = input['name']
                edge = [edge for edge in _edges if edge['sourceHandle'] == input_name]
                assert len(edge) == 1, f"Edge {input_name} not found"
                input = edge[0]['target']
                args += [input]
            graph[node['id']] = (fn, *args)
        evaluate = [node['id'] for node in body.nodes if evaluable(node['data'])]
        # print("\ngraph", graph)
        # print("\nevaluate", evaluate)
        # run graph
        results = get(graph, evaluate)
        # print('\nresults', results)
        return results
    except Exception as e:
        print('ERROR', repr(e))
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=repr(e))
    
# TODO: move functions to a separate file

def nodeId2Function(id):
    if id == "Area of Interest":
        return read_aoi
    if id == "Date Selector":
        return parse_dates
    if id == "Forest Monitoring":
        return forest_monitoring
    raise ValueError(f"Node {id} not found")

def evaluable(data): # should return serializable data
    if data['id'] == "Forest Monitoring":
        return True
    return False

def read_aoi(name):
    if name:
        return storage.read(f"{name}.geojson").__geo_interface__
    raise ValueError("AOI name not valid")

def parse_dates(dates):
    print("dates", dates)
    return dates

def forest_monitoring(aoi, dates):
    print("aoi", aoi)
    print("dates", dates)
    return "forest monitoring"
    
# need this to run in background
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action='store_true', help="Enable hot reloading")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
    # uvicorn.run("main:app", host=args.host, port=args.port, reload=True)
