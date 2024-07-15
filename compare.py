import os
import re
import nltk
import numpy as np
from nltk import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def read_text_files(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    texts = []
    for file in files:
        with open(os.path.join(directory, file), 'r', encoding='utf-8') as f:
            text = f.read()
            texts.append(text)
    return texts

def highlight_sentences(texts):
    # Tokenize sentences
    sentences = []
    for text in texts:
        sentences.extend(sent_tokenize(text))
        
    # Vectorize sentences using TF-IDF
    vectorizer = TfidfVectorizer(tokenizer=word_tokenize)
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Calculate pairwise cosine similarity between sentences
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Generate HTML with color-coded sentences based on similarity
    html_output = "<html><body>"
    
    # Centered header with red text color
    html_output += '<h1 style="text-align: center; color: rgb(255, 191, 191);">Text Files Similarity Tool</h1>'
    
    # Color-coded sentences
    for i, sentence in enumerate(sentences):
        similarity_scores = similarity_matrix[i]
        color_intensity = int(np.mean(similarity_scores) * 255)
        color = f"rgb({255}, {255-color_intensity}, {255-color_intensity})"  # Red-Green-Blue color format
        highlighted_sentence = f'<span style="background-color:{color};">{sentence}</span><br>'
        html_output += highlighted_sentence
    
    html_output += "</body></html>"

    # Table for displaying sentence similarities
    html_output += "<html><body><hr>"
    html_output += "<center><h2>Sentence Similarities</h2></center>"
    html_output += "<table style='width:100%; border-collapse: collapse; border: 1px solid black;'>"
    html_output += "<tr><th style='border: 1px solid black; padding: 8px;'>Sentence</th><th style='border: 1px solid black; padding: 8px;'>Most Similar Sentence</th><th style='border: 1px solid black; padding: 8px;'>Second Most Similar Sentence</th></tr>"
    
    for i, sentence in enumerate(sentences):
        similarity_scores = similarity_matrix[i]
        # Get indices of top 2 most similar sentences (excluding itself)
        most_similar_indices = np.argsort(similarity_scores)[-3:-1][::-1]
        most_similar_sentence = sentences[most_similar_indices[0]]
        second_most_similar_sentence = sentences[most_similar_indices[1]]
        
        # Add borders to each cell in the table
        html_output += f"<tr><td style='border: 1px solid black; padding: 8px;'>{sentence}</td><td style='border: 1px solid black; padding: 8px;'>{most_similar_sentence}</td><td style='border: 1px solid black; padding: 8px;'>{second_most_similar_sentence}</td></tr>"
    
    html_output += "</table>"
    html_output += "</body></html>"

    return html_output

# Example usage
if __name__ == "__main__":
    directory = './'  # Directory containing text files
    texts = read_text_files(directory)
    html_result = highlight_sentences(texts)

    # Save or display the HTML result
    with open('highlighted_sentences.html', 'w', encoding='utf-8') as f:
        f.write(html_result)
    print("HTML file 'highlighted_sentences.html' generated successfully.")
