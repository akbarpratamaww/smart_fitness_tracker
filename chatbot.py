import openai
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client (optional - falls back to rule-based if no API key)
openai_client = None
if os.getenv('OPENAI_API_KEY'):
    openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class FitnessChatbot:
    def __init__(self, user_data=None):
        self.user_data = user_data
        self.conversation_history = []
        
    def get_response(self, user_message, context=None):
        """Generate response using OpenAI GPT (or rule-based fallback)."""
        
        # Build context prompt
        context_prompt = self._build_context_prompt()
        
        if openai_client:
            return self._get_ai_response(user_message, context_prompt)
        else:
            return self._get_rule_based_response(user_message, context)
    
    def _build_context_prompt(self):
        """Build context prompt from user data."""
        if not self.user_data:
            return ""
        
        return f"""
        User Profile:
        - Age: {self.user_data.get('age', 'N/A')}
        - Gender: {self.user_data.get('gender', 'N/A')}
        - Height: {self.user_data.get('height_cm', 'N/A')} cm
        - Weight: {self.user_data.get('weight_kg', 'N/A')} kg
        - BMR: {self.user_data.get('bmr', 'N/A')} kcal/day
        - TDEE: {self.user_data.get('tdee', 'N/A')} kcal/day
        - Daily Target: {self.user_data.get('daily_target_calories', 'N/A')} kcal
        - Fitness Goal: {self.user_data.get('fitness_goal', 'N/A')}
        
        You are a friendly, knowledgeable fitness and nutrition coach. 
        Provide concise, actionable advice. Keep responses under 200 words.
        """
    
    def _get_ai_response(self, user_message, context_prompt):
        """Get response from OpenAI API."""
        try:
            messages = [
                {"role": "system", "content": f"You are an expert fitness and nutrition coach. {context_prompt}"},
                {"role": "user", "content": user_message}
            ]
            
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"I'm having trouble connecting to my AI service. Please try again later. (Error: {str(e)})"
    
    def _get_rule_based_response(self, user_message, context):
        """Fallback rule-based responses when AI is unavailable."""
        user_message_lower = user_message.lower()
        
        # Fitness goal advice
        if any(word in user_message_lower for word in ['lose weight', 'weight loss', 'fat loss']):
            return """For weight loss, focus on:
1. Calorie deficit: Eat 300-500 calories below your TDEE
2. Increase protein intake to preserve muscle
3. Combine strength training with cardio
4. Aim for 7,000-10,000 steps daily
Would you like specific meal or workout recommendations?"""
        
        elif any(word in user_message_lower for word in ['gain muscle', 'build muscle', 'muscle gain']):
            return """For muscle gain:
1. Eat in a calorie surplus (200-300 above TDEE)
2. Consume 1.6-2.2g protein per kg body weight
3. Focus on progressive overload in strength training
4. Get adequate sleep for recovery (7-9 hours)
Want me to create a sample workout split?"""
        
        elif any(word in user_message_lower for word in ['workout', 'exercise', 'training']):
            return """Great! What's your experience level and available equipment?
I can recommend:
- Bodyweight workouts (no equipment)
- Dumbbell/kettlebell routines
- Full gym splits
- Home cardio sessions
Let me know what you prefer!"""
        
        elif any(word in user_message_lower for word in ['meal', 'food', 'diet', 'eat']):
            return """For healthy eating:
- Prioritize whole foods: lean proteins, complex carbs, healthy fats
- Eat protein with every meal
- Include colorful vegetables
- Stay hydrated (2-3L water daily)
- Plan meals ahead to avoid impulse choices
Would you like a sample meal plan?"""
        
        elif any(word in user_message_lower for word in ['calorie', 'calories', 'burn']):
            if context and 'calories_in' in context and 'calories_out' in context:
                return f"""Today's calorie summary:
Consumed: {context['calories_in']} kcal
Burned: {context['calories_out']} kcal
Balance: {context['calories_in'] - context['calories_out']} kcal

Tip: Focus on whole foods and stay active to maintain your target."""
            return """To manage calories effectively:
1. Track everything you eat and drink
2. Use smaller plates for portion control
3. Eat slowly and mindfully
4. Choose water over sugary drinks
5. Include protein and fiber at every meal"""
        
        elif any(word in user_message_lower for word in ['motivation', 'motivate', 'stuck', 'give up']):
            return """Remember your 'why'! Progress takes time and consistency.
✅ Celebrate small wins
✅ Track your progress (photos, measurements, strength gains)
✅ Find an accountability partner
✅ Visualize your success
You've got this! 💪 What's one small action you can take today?"""
        
        elif any(word in user_message_lower for word in ['sleep', 'rest', 'recovery']):
            return """Sleep and recovery are crucial for fitness:
- Aim for 7-9 hours quality sleep
- Take rest days to prevent injury
- Active recovery: walking, stretching, yoga
- Listen to your body's signals
Need tips for better sleep?"""
        
        elif any(word in user_message_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']):
            return f"""Hello! 👋 I'm your fitness assistant. 
{self._get_daily_greeting()}
How can I help you with your fitness journey today? I can help with:
• Nutrition advice and meal planning
• Workout recommendations
• Calorie tracking guidance
• Motivation and tips
• Answering fitness questions"""
        
        else:
            return f"""Thanks for your message! I'm here to help with your fitness journey.

Here are some things I can assist with:
• Weight loss or muscle gain strategies
• Workout plans for any equipment level
• Healthy meal ideas and recipes
• Calorie tracking guidance
• Motivation and consistency tips

What would you like to focus on today? 😊"""
    
    def _get_daily_greeting(self):
        """Generate a friendly daily greeting."""
        from datetime import datetime
        hour = datetime.now().hour
        
        if hour < 12:
            return "Good morning! Ready to make today a great day for your health?"
        elif hour < 18:
            return "Good afternoon! How's your day going so far?"
        else:
            return "Good evening! How did your fitness goals go today?"