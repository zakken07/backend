import os
import base64
import json
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import google.generativeai as genai
from io import BytesIO
from PIL import Image
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="FoodScan AI API",
    description="API untuk menganalisis gambar makanan menggunakan Gemini AI",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Pydantic model for request
class ImageAnalysisRequest(BaseModel):
    image: str  # Base64 encoded image
    mime_type: Optional[str] = "image/jpeg"

# Pydantic model for response
class NutritionInfo(BaseModel):
    protein: str
    carbs: str
    fat: str
    fiber: str

class AnalysisResponse(BaseModel):
    food_name: str
    freshness_level: str
    freshness_score: int
    estimated_calories: int
    nutrition_summary: NutritionInfo
    analysis_summary: str
    recommendations: list[str]

# Configure Gemini API
def configure_gemini():
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

# Helper function to decode and process image
def process_image(image_data: str, mime_type: str) -> Image.Image:
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        
        # Open with PIL
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize if too large (max 1024x1024)
        max_size = 1024
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        return image
        
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

# Helper function to parse Gemini response
def parse_gemini_response(response_text: str) -> Dict[str, Any]:
    try:
        # Try to extract JSON from response
        if '```json' in response_text:
            json_start = response_text.find('```json') + 7
            json_end = response_text.find('```', json_start)
            json_str = response_text[json_start:json_end].strip()
        else:
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
            else:
                json_str = response_text
        
        # Parse JSON
        result = json.loads(json_str)
        
        # Ensure required fields exist
        default_response = {
            "food_name": result.get("food_name", "Unknown food"),
            "freshness_level": result.get("freshness_level", "Cannot determine"),
            "freshness_score": result.get("freshness_score", 50),
            "estimated_calories": result.get("estimated_calories", 0),
            "nutrition_summary": result.get("nutrition_summary", {
                "protein": "N/A",
                "carbs": "N/A", 
                "fat": "N/A",
                "fiber": "N/A"
            }),
            "analysis_summary": result.get("analysis_summary", response_text),
            "recommendations": result.get("recommendations", ["Try taking a photo with better lighting"])
        }
        
        return default_response
        
    except json.JSONDecodeError:
        # Fallback response if JSON parsing fails
        return {
            "food_name": "Unknown food",
            "freshness_level": "Cannot determine",
            "freshness_score": 50,
            "estimated_calories": 0,
            "nutrition_summary": {
                "protein": "N/A",
                "carbs": "N/A",
                "fat": "N/A", 
                "fiber": "N/A"
            },
            "analysis_summary": response_text,
            "recommendations": ["Try taking a photo with better lighting"]
        }

# Health check endpoint
@app.get("/")
async def root():
    return {
        "message": "FoodScan AI API is running!",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "foodscan-ai"}

# Main analysis endpoint
@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_food_image(request: ImageAnalysisRequest):
    try:
        # Configure Gemini
        model = configure_gemini()
        
        # Process image
        image = process_image(request.image, request.mime_type)
        
        # Convert image to bytes for Gemini
        buffer = BytesIO()
        image.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        
        # Comprehensive prompt for food analysis
        prompt = """
        Analisis gambar makanan ini dan berikan hasil dalam format JSON yang valid dengan struktur berikut:
        
        {
            "food_name": "nama spesifik makanan",
            "freshness_level": "segar/menengah/tidak segar",
            "freshness_score": 85,
            "estimated_calories": 250,
            "nutrition_summary": {
                "protein": "15g",
                "carbs": "30g", 
                "fat": "8g",
                "fiber": "3g"
            },
            "analysis_summary": "ringkasan detail analisis gizi dan kesegaran makanan",
            "recommendations": ["rekomendasi 1", "rekomendasi 2"]
        }
        
        Petunjuk analisis:
        1. Identifikasi jenis makanan se-spesifik mungkin
        2. Evaluasi tingkat kesegaran (0-100): segar (80-100), menengah (50-79), tidak segar (0-49)
        3. Estimasi kalori berdasarkan porsi dan jenis makanan
        4. Analisis kandungan gizi dasar (protein, karbohidrat, lemak, serat)
        5. Berikan ringkasan analisis yang informatif
        6. Berikan 2-3 rekomendasi yang berguna
        
        Pastikan response adalah JSON yang valid tanpa karakter tambahan.
        """
        
        # Generate content with Gemini
        response = model.generate_content([
            prompt,
            {"mime_type": "image/jpeg", "data": image_bytes}
        ])
        
        # Parse response
        result = parse_gemini_response(response.text)
        
        return AnalysisResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")

# Vercel serverless function handler
def handler(request):
    """
    Handler for Vercel serverless functions
    """
    return app(request)

# For local development
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
