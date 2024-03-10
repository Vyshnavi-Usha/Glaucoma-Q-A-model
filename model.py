import nltk
nltk.download('punkt')
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
import nltk
#nltk.download('punkt')
from sklearn.metrics import classification_report, f1_score

def load_qa_pairs_from_csv(file_path):
    questions = []
    answers = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            questions.append(row[0])
            answers.append(row[1])
    return questions, answers

file_path = "/content/question_and_answer.csv"
questions, answers = load_qa_pairs_from_csv(file_path)

X_train, X_test, y_train, y_test = train_test_split(questions, answers, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', SVC(kernel='linear'))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
#print(classification_report(y_test, y_pred))

references = [word_tokenize(answer.lower()) for answer in y_test]
candidates = [word_tokenize(answer.lower()) for answer in y_pred]

print("Welcome to the GlaucomaCataractBot! You can start chatting with the bot. Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
    else:
        response = pipeline.predict([user_input])
        print("GlaucomaCataractBot:", response[0])