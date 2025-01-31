# Celebrity-Info

This Streamlit-based web app provides detailed information about celebrities, including their name, date of birth, and major historical events that occurred around their birthdate. The app integrates with powerful language models like GroqCloud's API to fetch and summarize celebrity details, and uses APIs like Wikipedia to fetch relevant data like images and biographical summaries.

![Demo](https://github.com/Carnage203/Celebrity-Info/blob/f8bba3f778b5368981bc57124620c8008d0cd1c3/example1.png)

# Features
Celebrity Search: Enter the name of a celebrity to retrieve information.  
Biography: Get a brief biography and additional facts about the celebrity.  
Date of Birth: Discover the date of birth of the celebrity.  
Historical Events: Get a list of major global events that occurred around the celebrity's birthdate.  
Celebrity Image: Fetch an image of the celebrity (using Wikipedia API or other alternatives like Unsplash) to display alongside their information.  
Interactive UI: All information is displayed in a neat, user-friendly, and interactive interface with expandable sections (cards) for each type of information.  

# Technologies Used
Streamlit: For building the interactive web interface.   
GroqCloud's API: For querying and generating detailed text about celebrities.  
Wikipedia API: To retrieve celebrity images and additional facts.  
Requests: For making API requests to external services.  
