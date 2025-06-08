# From https://datastax.github.io/graph-rag/examples/movie-reviews-graph-rag/#the-strategy

from dotenv import load_dotenv

# load environment variables from the .env file
load_dotenv()

import pandas as pd
from io import StringIO

reviews_data_string = """
id,reviewId,creationDate,criticName,isTopCritic,originalScore,reviewState,publicatioName,reviewText,scoreSentiment,reviewUrl
addams_family,2644238,2019-11-10,James Kendrick,False,3/4,fresh,Q Network Film Desk,captures the family's droll humor with just the right mixture of morbidity and genuine care,POSITIVE,http://www.qnetwork.com/review/4178
addams_family,2509777,2018-09-12,John Ferguson,False,4/5,fresh,Radio Times,A witty family comedy that has enough sly humour to keep adults chuckling throughout.,POSITIVE,https://www.radiotimes.com/film/fj8hmt/the-addams-family/
addams_family,26216,2000-01-01,Rita Kempley,True,,fresh,Washington Post,"More than merely a sequel of the TV series, the film is a compendium of paterfamilias Charles Addams's macabre drawings, a resurrection of the cartoonist's body of work. For family friends, it would seem a viewing is de rigueur mortis.",POSITIVE,http://www.washingtonpost.com/wp-srv/style/longterm/movies/videos/theaddamsfamilypg13kempley_a0a280.htm
the_addams_family_2019,2699537,2020-06-27,Damond Fudge,False,,fresh,"KCCI (Des Moines, IA)","As was proven by the 1992-93 cartoon series, animation is the perfect medium for this creepy, kooky family, allowing more outlandish escapades",POSITIVE,https://www.kcci.com/article/movie-review-the-addams-family/29443537
the_addams_family_2019,2662133,2020-01-21,Ryan Silberstein,False,,fresh,Cinema76,"This origin casts the Addams family as an immigrant story, and the film leans so hard into the theme of accepting those different from us and valuing diversity over conformity,",POSITIVE,https://www.cinema76.com/home/2019/10/11/the-addams-family-is-a-fun-update-to-an-iconic-american-clan
the_addams_family_2019,2661356,2020-01-17,Jennifer Heaton,False,5.5/10,rotten,Alternative Lens,...The film's simplistic and episodic plot put a major dampener on what could have been a welcome breath of fresh air for family animation.,NEGATIVE,https://altfilmlens.wordpress.com/2020/01/17/my-end-of-year-surplus-review-extravaganza-thing-2019/
the_addams_family_2,102657551,2022-02-16,Mat Brunet,False,4/10,rotten,AniMat's Review (YouTube),The Addams Family 2 repeats what the first movie accomplished by taking the popular family and turning them into one of the most boringly generic kids films in recent years.,NEGATIVE,https://www.youtube.com/watch?v=G9deslxPDwI
the_addams_family_2,2832101,2021-10-15,Sandie Angulo Chen,False,3/5,fresh,Common Sense Media,This serviceable animated sequel focuses on Wednesday's feelings of alienation and benefits from the family's kid-friendly jokes and road trip adventures.,POSITIVE,https://www.commonsensemedia.org/movie-reviews/the-addams-family-2
the_addams_family_2,2829939,2021-10-08,Emily Breen,False,2/5,rotten,HeyUGuys,"Lifeless and flat, doing a disservice to the family name and the talent who voice them. WIthout glamour, wit or a hint of a soul. A void. Avoid.",NEGATIVE,https://www.heyuguys.com/the-addams-family-2-review/
addams_family_values,102735159,2022-09-22,Sean P. Means,False,3/4,fresh,Salt Lake Tribune,Addams Family Values is a ghoulishly fun time. It would have been a real howl if the producers weren't too scared to go out on a limb in this twisted family tree.,POSITIVE,https://www.newspapers.com/clip/110004014/addams-family-values/
addams_family_values,102734540,2022-09-21,Jami Bernard,True,3.5/4,fresh,New York Daily News,"The title is apt. Using those morbidly sensual cartoon characters as pawns, the new movie Addams Family Values launches a witty assault on those with fixed ideas about what constitutes a loving family. ",POSITIVE,https://www.newspapers.com/clip/109964753/addams-family-values/
addams_family_values,102734521,2022-09-21,Jeff Simon,False,3/4,fresh,Buffalo News,"Addams Family Values has its moments -- rather a lot of them, in fact. You knew that just from the title, which is a nice way of turning Charles Addams' family of ghouls, monsters and vampires loose on Dan Quayle.",POSITIVE,https://buffalonews.com/news/quirky-values-the-addams-family-returns-with-a-bouncing-baby/article_2aafde74-da6c-5fa7-924a-76bb1a906d9c.html
"""

movies_data_string = """
id,title,audienceScore,tomatoMeter,rating,ratingContents,releaseDateTheaters,releaseDateStreaming,runtimeMinutes,genre,originalLanguage,director,writer,boxOffice,distributor,soundMix
addams_family,The Addams Family,66,67,,,1991-11-22,2005-08-18,99,Comedy,English,Barry Sonnenfeld,"Charles Addams,Caroline Thompson,Larry Wilson",$111.3M,Paramount Pictures,"Surround, Dolby SR"
the_addams_family_2019,The Addams Family,69,45,PG,"['Some Action', 'Macabre and Suggestive Humor']",2019-10-11,2019-10-11,87,"Kids & family, Comedy, Animation",English,"Conrad Vernon,Greg Tiernan","Matt Lieberman,Erica Rivinoja",$673.0K,Metro-Goldwyn-Mayer,Dolby Atmos
the_addams_family_2,The Addams Family 2,69,28,PG,"['Macabre and Rude Humor', 'Language', 'Violence']",2021-10-01,2021-10-01,93,"Kids & family, Comedy, Adventure, Animation",English,"Greg Tiernan,Conrad Vernon","Dan Hernandez,Benji Samit,Ben Queen,Susanna Fogel",$56.5M,Metro-Goldwyn-Mayer,
addams_family_reunion,Addams Family Reunion,33,,,,,,92,Comedy,English,Dave Payne,,,,
addams_family_values,Addams Family Values,63,75,,,1993-11-19,2003-08-05,93,Comedy,English,Barry Sonnenfeld,Paul Rudnick,$45.7M,"Argentina Video Home, Paramount Pictures","Surround, Dolby Digital"
"""

reviews_all = pd.read_csv(StringIO(reviews_data_string))
movies_all = pd.read_csv(StringIO(movies_data_string))

reviews_data = reviews_all.rename(columns={"id": "reviewed_movie_id"})
movies_data = movies_all.rename(columns={"id": "movie_id"})

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

# create the vector store
vectorstore = InMemoryVectorStore(OpenAIEmbeddings())

from langchain_core.documents import Document

documents = []
# convert each movie into a LangChain document
for index, row in movies_data.iterrows():
    content = str(row["title"])
    metadata = row.fillna("").astype(str).to_dict()
    metadata["doc_type"] = "movie_info"
    document = Document(page_content=content, metadata=metadata)
    documents.append(document)

# Convert each movie review into a LangChain document
for index, row in reviews_data.iterrows():
    content = str(row["reviewText"])
    metadata = row.drop("reviewText").fillna("").astype(str).to_dict()
    metadata["doc_type"] = "movie_review"
    document = Document(page_content=content, metadata=metadata)
    documents.append(document)


# check the total number of documents
print("There are", len(documents), "total Documents")

vectorstore.add_documents(documents)


from graph_retriever.strategies import Eager
from langchain_graph_retriever import GraphRetriever

retriever = GraphRetriever(
    store=vectorstore,
    edges=[("reviewed_movie_id", "movie_id")],
    strategy=Eager(start_k=10, adjacent_k=10, select_k=100, max_depth=1),
)

INITIAL_PROMPT_TEXT = "What are some good family movies?"
# INITIAL_PROMPT_TEXT = "What are some recommendations of exciting action movies?"
# INITIAL_PROMPT_TEXT = "What are some classic movies with amazing cinematography?"


# invoke the query
query_results = retriever.invoke(INITIAL_PROMPT_TEXT)

# print the raw retrieved results
for result in query_results:
    print(result.metadata["doc_type"], ": ", result.page_content)
    print(result.metadata)
    print()


# collect the movie info for each film retrieved
compiled_results = {}
for result in query_results:
    if result.metadata["doc_type"] == "movie_info":
        movie_id = result.metadata["movie_id"]
        movie_title = result.metadata["title"]
        compiled_results[movie_id] = {
            "movie_id": movie_id,
            "movie_title": movie_title,
            "reviews": {},
        }

# go through the results a second time, collecting the retreived reviews for
# each of the movies
for result in query_results:
    if result.metadata["doc_type"] == "movie_review":
        reviewed_movie_id = result.metadata["reviewed_movie_id"]
        review_id = result.metadata["reviewId"]
        review_text = result.page_content
        compiled_results[reviewed_movie_id]["reviews"][review_id] = review_text

formatted_text = ""
for movie_id, review_list in compiled_results.items():
    formatted_text += "\n\n Movie Title: "
    formatted_text += review_list["movie_title"]
    formatted_text += "\n Movie ID: "
    formatted_text += review_list["movie_id"]
    for review_id, review_text in review_list["reviews"].items():
        formatted_text += "\n Review: "
        formatted_text += review_text


print(formatted_text)