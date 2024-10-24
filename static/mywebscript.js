function RunSentimentAnalysis() {
    const text = document.getElementById('textToAnalyze').value;
    fetch(`/emotionDetector?text=${encodeURIComponent(text)}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok.');
            }
            return response.json();
        })
        .then(data => {
            document.getElementById('system_response').innerText = JSON.stringify(data, null, 2);
        })
        .catch(error => console.error('Error:', error));
}
