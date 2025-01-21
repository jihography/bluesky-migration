def classify_countries_batch(batch_texts, batch_indices):
    while True:
        try:
            # Create the JSON-based prompt
            prompt = '''
                You are an AI trained to classify text based on its content. 
                Identify the country or countries associated with the migration issue being discussed. 
                Determine the relevant country or countries based on specific issues, events, names, or place names mentioned in the text.

                Migration is defined as "the movement of people away from their usual place of residence to a new place of residence, either across an international border or within a state."

                If multiple countries are mentioned, provide the answer as a list separated by commas (e.g., "US, MX"). 
                However, if no specific country can be identified from the text or if the text is not related to human migration, respond with "XXX."

                Additional clarification:
                - The nationality of the person writing or speaking is not relevant. For instance, a German person could discuss migration issues in the United States in German. The focus is solely on the content of the text.
                - The text must pertain to human migration. For example, a description of the landscape near a border without any connection to migration would not qualify.


                Your response must strictly follow this JSON format:
                {
                    "results": [
                        {"index": <index>, "country": "<country_code>"},
                        {"index": <index>, "country": "<country_code>"},
                        ...
                    ]
                }

                Here are example inputs and responses:

                ## Input ##
                {
                    "inputs": [
                        {"index": 0, "text": "Riots have erupted across the U.K. after the murder of 3 girls in England last week. MSM calls protesters 'far right thugs.' But others believe Britain's migrant crisis has reached a boiling point."},
                        {"index": 1, "text": "A brazilian migrant killed 14 year old Daniel Anjorin 3 months ago."},
                        {"index": 2, "text": "Actually, Biden ended the policy of family detention shortly after taking office, which was used to detain migrant families, including children. He ended the zero tolerance policy of family detention in December 2021."},
                        {"index": 3, "text": "\"A vast network of human trafficking.\" Russia fraudulently recruits foreigners from the poorest countries for war. The fate of migrant workers who came to the Russian Federation legally and voluntarily is no better."},
                        {"index": 4, "text": "Israeli Defense Minister Gallant: \"We have stabilized the border, we have launched an offensive. We will wipe Hamas off the face of the earth.\""},
                        {"index": 5, "text": "I live on Vancouver ISLAND, and it’s 12 hours to the North tip. But from the Canadian border in BC to the border of the Yukon, that’s 24 hours of driving — no breaks. Not even our biggest province tho. Texas has nothing on Canada. :)"},
                        {"index": 6, "text": "All these idiot elected Dems being confused by the GOP not taking yes on the border bill is so tiresome. You can't out racism the extremely racist party. Trying to give in to them is pointless. The Dems are the frog and the GOP is the scorpion. The scorpion stings. Do better, Dems."},
                        {"index": 7, "text": "A short & very useful summary of the EU migration pact saga by Simon @Usherwood #eumigrationpact :globe_with_meridians:"},
                        {"index": 8, "text": "So it’s entirely possible that when all is said and done, President Biden will get additional funding for Ukraine and Israel and have made zero concessions on the border to get it done. If that happens … it would be impressive. ..."},
                        {"index": 9, "text": "Spürt ihr schon den Wohlstand für alle, wenn wir endlich die Migration begrenzen und schneller abschieben?"},
                        {"index": 10, "text": "happy Black History Month yall! In the spirit of things I was reading up about some of my own history. My father is a Jamaican-born immigrant who moved to NYC in the mid-'70s. His grandmother is half-Chinese, and turns out there's some interesting history here: ..."},
                        {"index": 11, "text": "I swear over on Twitter Spaces RFK Jr just said Israel has border problems with its African population similar to the US's with Mexico. I think he misspoke in his haste show Musk how much he loves apartheid in all its forms."},
                        {"index": 12, "text": "He tried to apply for asylum in Switzerland, denied on procedural grounds. He went to Denmark, where he witnessed a suicide attempt of another refugee, denied. Returned to the Netherlands — and now told \"you can't describe your feelings to men good enough\", denied."},
                        {"index": 13, "text": "Ernesto Castañeda-Tinoco, director of the Immigration Lab, said \"there is no evidence of members of Hamas in Mexico preparing attacks on the U.S. The geopolitical situation at the U.S.-Mexico border is different.\""}
                    ]
                }

                ## Response ##
                {
                    "results": [
                        {"index": 0, "country": "GB"},
                        {"index": 1, "country": "BR, GB"},
                        {"index": 2, "country": "US"},
                        {"index": 3, "country": "RU"},
                        {"index": 4, "country": "XXX"},
                        {"index": 5, "country": "XXX"},
                        {"index": 6, "country": "US"},
                        {"index": 7, "country": "XXX"},
                        {"index": 8, "country": "US"},
                        {"index": 9, "country": "XXX"},
                        {"index": 10, "country": "US, JM"},
                        {"index": 11, "country": "US, MX, IL"},
                        {"index": 12, "country": "CH, DK, NL"},
                        {"index": 13, "country": "US, MX, PS"}
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
            response_content = re.sub(r"^```json|```$", "", response.choices[0].message.content).strip()

            # Validate the response is JSON
            try:
                response_json = json.loads(response_content)
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}. Response content: {response_content}")
                raise ValueError("Failed to parse JSON from the API response.")

            # Extract classifications
            classifications = {item['index']: item['country'] for item in response_json['results']}
            return classifications

        except Exception as e:
            print(f"An error occurred: {e}. Retrying...")
            time.sleep(5)  # Retry after a short delay

openai.api_key = 'api-key'

# Load the dataset
df = pd.read_csv("./Data/checkpoint_250117.csv")
# df = df.head(100)

# Add an empty 'country' column for results
if 'country' not in df.columns:
    df['country'] = None

output_file = "checkpoint_250117_classified.csv"

# Process the DataFrame in batches
batch_size = 500

# Batches for logging errors
error_batches = []

for start_idx in range(0, len(df), batch_size):
    end_idx = min(start_idx + batch_size, len(df))
    batch_texts = df.loc[start_idx:end_idx - 1, 'text'].tolist()
    batch_indices = df.loc[start_idx:end_idx - 1].index.tolist()
    print(f"Processing batch {start_idx} to {end_idx - 1}...")

    try:
        # Apply function
        batch_results = classify_countries_batch(batch_texts, batch_indices)

        # Update dataframe
        for idx, country in batch_results.items():
            df.at[idx, 'country'] = country

        # Save to CSV file
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"Completed processing batch {start_idx} to {end_idx - 1}!! Saved progress to {output_file}.")

    except Exception as e:
        # Log error ocurred batch
        print(f"An error occurred while processing batch {start_idx} to {end_idx - 1}: {e}")
        print("Skipping this batch and continuing with the next one.")
        error_batches.append((start_idx, end_idx - 1))
        continue 

    time.sleep(2) 

# Print error log
if error_batches:
    print("\nThe following batches encountered errors and were skipped:")
    for batch in error_batches:
        print(f"Batch {batch[0]} to {batch[1]}")
else:
    print("\nAll batches appended successfully.")

print("Processing complete. Results saved.")
