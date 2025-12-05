import google.generativeai as genai
import base64
import json
import os
from http.server import BaseHTTPRequestHandler

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # Get content length
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Parse JSON data
            data = json.loads(post_data.decode('utf-8'))
            
            # Get image data
            image_data = data.get('image')
            if not image_data:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'No image data provided'}).encode())
                return
            
            # Configure Gemini API
            api_key = os.environ.get('GEMINI_API_KEY')
            if not api_key:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Gemini API key not configured'}).encode())
                return
            
            genai.configure(api_key=api_key)
            
            # Initialize Gemini model
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_data.split(',')[1])
            
            # Prepare the prompt for food analysis
            prompt = """
            Analisis gambar makanan ini dan berikan hasil dalam format JSON dengan struktur berikut:
            {
                "food_name": "nama makanan",
                "freshness_level": "segar/menengah/tidak segar",
                "freshness_score": 85,
                "estimated_calories": 250,
                "nutrition_summary": {
                    "protein": "15g",
                    "carbs": "30g",
                    "fat": "8g",
                    "fiber": "3g"
                },
                "analysis_summary": "ringkasan analisis gizi dan kesegaran",
                "recommendations": ["rekomendasi 1", "rekomendasi 2"]
            }
            
            Fokus pada:
            1. Identifikasi jenis makanan
            2. Estimasi tingkat kesegaran (0-100)
            3. Perkiraan kalori
            4. Analisis gizi dasar
            5. Ringkasan dan rekomendasi
            """
            
            # Generate content
            response = model.generate_content([
                prompt,
                {"mime_type": "image/jpeg", "data": image_bytes}
            ])
            
            # Extract JSON from response
            response_text = response.text
            # Clean up response to extract JSON
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
            
            # Parse and validate JSON
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                result = {
                    "food_name": "Unknown",
                    "freshness_level": "tidak dapat ditentukan",
                    "freshness_score": 50,
                    "estimated_calories": 0,
                    "nutrition_summary": {
                        "protein": "N/A",
                        "carbs": "N/A",
                        "fat": "N/A",
                        "fiber": "N/A"
                    },
                    "analysis_summary": response_text,
                    "recommendations": ["Coba ambil foto dengan pencahayaan lebih baik"]
                }
            
            # Send successful response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
            
        except Exception as e:
            # Send error response
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())
    
    def do_OPTIONS(self):
        # Handle preflight requests
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()