def generate_hashtags_batch(batch_texts, batch_indices):
    while True:
        try:
            # Few-shot learning prompt for generating hashtags
            prompt = '''
                You are an AI that generates concise and relevant hashtags based on the given text.
                In addition to using words from the text, include words that capture the context or underlying meaning of the content.
                Identify key themes, topics, or ideas from the content and create up to 5 hashtags. 
                Provide the hashtags as a comma-separated list without the "#" symbol 
                (e.g., "Example1, Example2, Example3"). 
                Avoid using overly general words like 'Opinion', 'Article', or 'News'.
                Ensure each hashtag is capitalized, lemmatized and maintains spaces between words such as 'Border Control' (not 'BorderControl').
                However, 'Immigration/Immigrant' and 'Migration/Migrant' should not be lemmatized.
                Respond only in string format.

                Your response must strictly follow this JSON format:
                {
                    "results": [
                        {"index": <index>, "hashtags": "<hashtags>"},
                        {"index": <index>, "hashtags": "<hashtags>"},
                        ...
                    ]
                }

                Here are example inputs and responses:

                ## Input ##
                {
                    "inputs": [
                        {"index": 0, "text": "What Is Really Happening In The U.K. ?: Riots have erupted across the U.K. after the murder of 3 girls in England last week. MSM calls protesters 'far right thugs.' But others believe Britain's migrant crisis has reached a boiling point."},
                        {"index": 1, "text": "A brazilian migrant killed 14 year old Daniel Anjorin 3 months ago."},
                        {"index": 2, "text": "Actually, Biden ended the policy of family detention shortly after taking office, which was used to detain migrant families, including children. He ended the zero tolerance policy of family detention in December 2021."},
                        {"index": 3, "text": "\"A vast network of human trafficking.\" Russia fraudulently recruits foreigners from the poorest countries for war. The fate of migrant workers who came to the Russian Federation legally and voluntarily is no better."},
                        {"index": 4, "text": "All these idiot elected Dems being confused by the GOP not taking yes on the border bill is so tiresome. You can't out racism the extremely racist party. Trying to give in to them is pointless. The Dems are the frog and the GOP is the scorpion. The scorpion stings. Do better, Dems."}
                    ]
                }

                ## Response ##
                {
                    "results": [
                        {"index": 0, "hashtags": "Immigration, Riots, UK, Murder, Crisis"},
                        {"index": 1, "hashtags": "Migrant, Murder, Brazil"},
                        {"index": 2, "hashtags": "Biden, Politics, Immigration, Family Detention, Policy"},
                        {"index": 3, "hashtags": "Russia, Human Rights, War, Migrant, Trafficking"},
                        {"index": 6, "hashtags": "Democrats, GOP, Border Bill, Racism, Politics"}
                    ]
                }

                Now process the following inputs:
                '''

            # Add input texts with indices in JSON format
            for idx, text in zip(batch_indices, batch_texts):
                prompt += f"{{\"index\": {idx}, \"text\": \"{text}\"}},\n"

            # Close the JSON array in the prompt
            prompt += '''
            Note: Ensure your response strictly adheres to the JSON format provided above. 
            No additional text or characters should appear outside the JSON object.
            '''

            # Make the API call
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )

            # Parse the API response
            response_content = response.choices[0].message.content.strip()
            response_content = re.sub(r"^json|$", "", response.choices[0].message.content).strip()

            # Validate the response is JSON
            try:
                response_json = json.loads(response_content)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}. Response content: {response_content}")
                raise ValueError("Failed to parse JSON from the API response.")

            # Extract classifications
            classifications = {item['index']: item['hashtags'] for item in response_json['results']}
            return classifications

        except Exception as e:
            print(f"An error occurred: {e}. Retrying...")
            time.sleep(5)  # Retry after a short delay
