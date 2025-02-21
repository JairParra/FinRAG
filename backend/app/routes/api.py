from fastapi import APIRouter

router = APIRouter()

@router.get('/query')
def query():
    return {'response': 'Sample response'}
