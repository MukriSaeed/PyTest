import os
import time
from flask import Flask, render_template, request, jsonify
import anthropic
from pinecone import Pinecone
from voyageai import Client as VoyageAI
import html
import random

app = Flask(__name__)

# Initialize clients
anthropic_client = anthropic.Anthropic(api_key="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")
pinecone_client = Pinecone(api_key="f5e0f9cf-5048-47cb-a238-ea15d48882e4")
voyage_client = VoyageAI(api_key="pa-W1QI5-5P_5MIyhzaygsc5IvAlie0XnMs4GShyMi_7jg")

index_name = "englishvector"
index = pinecone_client.Index(index_name)

def retry_with_exponential_backoff(func, max_retries=5, initial_delay=1, max_delay=60):
    def wrapper(*args, **kwargs):
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except anthropic.InternalServerError as e:
                if attempt == max_retries - 1:
                    raise
                sleep_time = min(delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                time.sleep(sleep_time)
        return func(*args, **kwargs)
    return wrapper

@retry_with_exponential_backoff
def translate_to_english(text):
    start_time = time.time()
    message = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": f"Translate the following Kurdish Sorani text to English without any words that don't exist in the Kurdish text: {text}"}
        ]
    )
    end_time = time.time()
    return message.content[0].text, end_time - start_time

from flask import Response

@retry_with_exponential_backoff
def stream_translate_to_kurdish_sorani(text):
    def generate():
        try:
            # Create a streaming client request
            with anthropic_client.messages.stream(
                max_tokens=1000,
                messages=[{"role": "user", "content": f"Translate the following English text to Kurdish Sorani (Central Kurdish) dialect without adding any words that don't exist in the English text: {text}"}],
                model="claude-3-5-sonnet-20240620",
            ) as stream:
                for response_text in stream.text_stream:
                    yield response_text  # Yield each piece of the stream as it's received

                    print(stream.text_stream)
        except Exception as e:
            yield f"Error: {str(e)}"
    
    return generate()

def get_embedding(text):
    start_time = time.time()
    embedding = voyage_client.embed(text, model="voyage-large-2").embeddings[0]
    end_time = time.time()
    return embedding, end_time - start_time


def get_similar_articles(embedding, similarity_threshold=0.7):
    start_time = time.time()
    results = index.query(vector=embedding, top_k=3, include_metadata=True)
    filtered_results = [match for match in results.matches if match.score >= similarity_threshold]
    end_time = time.time()

    # Collect the URLs for the articles
    article_urls = []
    for result in filtered_results:
        if "url" in result.metadata:
            article_urls.append(result.metadata["url"])
    
    return filtered_results, article_urls, end_time - start_time

@retry_with_exponential_backoff

def generate_response_from_articles(articles, user_input, article_urls):
    start_time = time.time()
    combined_text = ""
    for article in articles:
        if "Text" in article.metadata:
            combined_text += article.metadata["Text"] + " "
        elif "text" in article.metadata:
            combined_text += article.metadata["text"] + " "
        else:
            combined_text += "Article content not available. "

    if not combined_text.strip():
        return "I'm sorry, but I couldn't find any relevant information to answer your question.", 0

    # Generate the message from articles
    message = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        system="You are a helpful assistant. Answer the user's question based only on the information provided in the following articles.",
        messages=[
            {"role": "user", "content": f"Articles: {combined_text}\n\nUser's question: {user_input}"}
        ]
    )
    end_time = time.time()

    # Include the article URLs
    url_list_html = "<br>".join([f"<a href='{url}'>{url}</a>" for url in article_urls])

    return message.content[0].text, url_list_html, end_time - start_time

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    timings = {}

    try:
        # Step 2: Translate Kurdish Sorani to English
        english_query, timings['translate_to_english'] = translate_to_english(user_input)
        
        # Step 3: Get embedding for the English query
        query_embedding, timings['get_embedding'] = get_embedding(english_query)
        
        # Step 4: Find similar articles and their URLs
        similar_articles, article_urls, timings['get_similar_articles'] = get_similar_articles(query_embedding)
        
        # Check if we have any articles that meet the similarity threshold
        if not similar_articles:
            return jsonify({
                'response': "ببوورە، من نەمتوانی هیچ زانیارییەکی پەیوەندیدار بدۆزمەوە بۆ وەڵامدانەوەی پرسیارەکەت.",
                'html_urls': '',
                'timings': timings
            })
        
        # Step 5 & 6: Generate response from articles in English
        english_response, html_urls, timings['generate_response'] = generate_response_from_articles(similar_articles, english_query, article_urls)
        
        # Step 7: Translate response to Kurdish Sorani using streaming
        stream_response = stream_translate_to_kurdish_sorani(english_response)
      
        # Return the streamed response along with the URLs as a Flask Response object
        return Response(stream_response, content_type='text/plain')

    except Exception as e:
        return jsonify({
            'response': f"ببوورە، هەڵەیەک ڕوویدا: {str(e)}",
            'html_urls': '',
            'timings': timings
        })

if __name__ == "__main__":
    app.run(debug=True)