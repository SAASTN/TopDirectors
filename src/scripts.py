"""
Scripts used to generate the TopDirectors mark-down report based on IMDB database.
"""

import os
import numpy as np
import pandas as pd
from tabulate import tabulate

# The minimum number of votes a title must have in order to be considered for finding a director's average rating.
minNumVotes = 5000
# The minimum average rating a title must have in order to be considered for finding a director's average rating.
minAverageRatings = 5.0
# The number of high-voted titles to consider when finding a director's average rating.
numHighVotedTitles = 8
# The path to the data folder, relative to the repository where the IMDB files are stored.
data_folder_path = "src\\data"
# The path to the temporary folder where the files will be created to speed up multiple runs.
temp_folder_path = ""


def crete_temp_folder():
    """
    creates a temporary folder to store data. The folder is created within
    the data_folder_path and is named based on the values of minNumVotes,
    minAverageRatings, and numHighVotedTitles. If the folder already exists,
    it will not create a new one. The function returns the path to the
    created or existing temporary folder.    Returns:
        temp_folder_path (str): The path to the temporary folder.
    """
    temp_folder_path = os.path.join(data_folder_path, "temp")
    temp_folder_path = os.path.join(
        temp_folder_path, f"{minNumVotes}-{minAverageRatings:0.1f}-{numHighVotedTitles}"
    )
    if ~os.path.exists(temp_folder_path):
        from pathlib import Path

        Path(temp_folder_path).mkdir(parents=True, exist_ok=True)
    return temp_folder_path


def read_data_file(file_name: str) -> pd.DataFrame:
    """
    Reads an IMDB's ctv file. It also stores the read data frame in a Parquet file.
    In next runs, it reads data directly from the Parquet file instead of parsing
    the ctv file again.
    """
    folder = file_name + ".tsv"
    folder_dir = os.path.join(data_folder_path, folder)
    csv_file = os.path.join(folder_dir, "data.tsv")
    output_file = os.path.join(data_folder_path, f"{folder}.parquet")
    if os.path.exists(output_file):
        return pd.read_parquet(output_file)

    dtypes = {
        "title.ratings": {
            "tconst": str,
            "averageRating": np.float32,
            "numVotes": np.int32,
        },
        "title.basics": str,
        "title.crew": str,
        "name.basics": str,
    }
    df = pd.read_csv(
        csv_file, delimiter="\t", dtype=dtypes[file_name], na_values=["\\N"]
    )

    # Convert DataFrame to Parquet file for faster retrive next times
    df.to_parquet(output_file)
    return df


def try_read_temp_file(file_name: str) -> pd.DataFrame:
    """
    Reads and returns a temporary file if it exists.
    """
    file_path = os.path.join(temp_folder_path, file_name) + ".parquet"
    if os.path.exists(file_path):
        return pd.read_parquet(file_path)
    return None


def save_temp_file(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    """
    Saves atemporary data frame in a Parquet file.
    """
    file_path = os.path.join(temp_folder_path, file_name) + ".parquet"
    df.to_parquet(file_path)


def get_high_voted_movies():
    """
    Retrieves high voted movies based on specified criteria. These titles are
    considered for finding a director's average rating.
    Returns:
        `high_voted_movies_df` (DataFrame): DataFrame containing high voted movies information.
    """
    file_name = "HighVotedMovies"
    df = try_read_temp_file(file_name)
    if df is not None:
        return df

    ratings_df = read_data_file("title.ratings")

    ratings_high_voted_df = ratings_df.loc[
        (ratings_df["numVotes"] > minNumVotes)
        & (ratings_df["averageRating"] > minAverageRatings)
    ]

    titles_df = read_data_file("title.basics")

    movie_titles_df = titles_df.loc[titles_df["titleType"] == "movie"]
    high_voted_movies_df = pd.merge(
        ratings_high_voted_df, movie_titles_df, on="tconst", how="inner"
    )
    titles_crew_df = read_data_file("title.crew")
    high_voted_movies_df = pd.merge(
        high_voted_movies_df, titles_crew_df, on="tconst", how="inner"
    )
    names_df = read_data_file("name.basics")
    high_voted_movies_df = pd.merge(
        high_voted_movies_df,
        names_df,
        left_on="directors",
        right_on="nconst",
        how="inner",
    )
    high_voted_movies_df = high_voted_movies_df[
        [
            "tconst",
            "averageRating",
            "numVotes",
            "primaryTitle",
            "originalTitle",
            "startYear",
            "runtimeMinutes",
            "genres",
            "nconst",
            "primaryName",
            "birthYear",
            "deathYear",
        ]
    ].copy()
    high_voted_movies_df.rename(columns={"primaryName": "director"}, inplace=True)
    save_temp_file(high_voted_movies_df, file_name)
    return high_voted_movies_df


def get_high_voted_directors():
    """
    Retrieves high voted directors and their top movies based on the minimum number
    of top movies a director must have in order to be considered a high voted
    director. It calculating average ratings and votes and merges the director
    information with the top movies information.
    Returns:
            high_voted_directors_df (DataFrame): DataFrame containing high voted directors information.
            high_voted_directors_top_movies_df (DataFrame): DataFrame containing top movies of high voted directors.
    """
    high_voted_directors_top_movies_file_name = "HighVotedDirectorsTopMovies"
    high_voted_directors_file_name = "HighVotedDirectors"
    high_voted_directors_top_movies_df = try_read_temp_file(
        high_voted_directors_top_movies_file_name
    )
    high_voted_directors_df = try_read_temp_file(high_voted_directors_file_name)
    if (high_voted_directors_top_movies_df is not None) and (
        high_voted_directors_df is not None
    ):
        return high_voted_directors_df, high_voted_directors_top_movies_df

    high_voted_movies_df = get_high_voted_movies()

    # Group by 'nconst' and filter groups with at least k rows
    df_active_directors_titles = high_voted_movies_df.groupby("nconst").filter(
        lambda x: len(x) >= numHighVotedTitles
    )

    # Sort the remaining groups by 'averageRating' in descending order
    sorted_groups = df_active_directors_titles.sort_values(
        by="averageRating", ascending=False
    )

    # Keep only the top k rows from each group
    high_voted_directors_top_movies_df = sorted_groups.groupby("nconst").head(
        numHighVotedTitles
    )

    # Step 1: Group the DataFrame by 'nconst' and calculate mean average rating
    grouped_df = high_voted_directors_top_movies_df.groupby("nconst").agg(
        meanAverageRating=("averageRating", "mean"),
        sumVotes=("numVotes", "sum"),
        meanVotes=("numVotes", "mean"),
        minVotes=("numVotes", "min"),
    )

    # Step 2: Sort the DataFrame by mean average rating in descending order
    sorted_df = grouped_df.sort_values(
        ["meanAverageRating", "sumVotes"], ascending=False
    )

    # Step 4: Keep only the 'director', 'meanAverageRating', and 'nconst' columns
    high_voted_directors_df = sorted_df.reset_index().merge(
        high_voted_directors_top_movies_df[
            ["director", "nconst", "birthYear", "deathYear"]
        ].drop_duplicates(),
        on="nconst",
    )[
        [
            "nconst",
            "director",
            "meanAverageRating",
            "sumVotes",
            "meanVotes",
            "minVotes",
            "birthYear",
            "deathYear",
        ]
    ]

    save_temp_file(
        high_voted_directors_top_movies_df, high_voted_directors_top_movies_file_name
    )
    save_temp_file(high_voted_directors_df, high_voted_directors_file_name)
    return high_voted_directors_df, high_voted_directors_top_movies_df


def get_directors_all_movies(
    directors_df: pd.DataFrame, directors_top_movies_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Retrieves all movies directed by the directors in the given DataFrame. It also checks if these movies are top rated.

    Parameters:
        directors_df (pd.DataFrame): A pandas DataFrame containing director data.
        directors_top_movies_df (pd.DataFrame): A pandas DataFrame containing top movies data.

    Returns:
        pd.DataFrame: A DataFrame containing all movies directed by the directors, along with their rating status.

    The function first tries to read a temporary file named "HighVotedDirectorsAllMovies". If this file exists, it returns its content.
    If the file does not exist, it reads several data files ("title.basics", "title.crew", "title.ratings"), merges them with the input
    DataFrames, filters out non-movie titles, and adds a column indicating whether the movie is a top-rated movie.
    Finally, it saves the resulting DataFrame to a temporary file and returns it.
    """
    file_name = "HighVotedDirectorsAllMovies"
    df = try_read_temp_file(file_name)
    if df is not None:
        return df
    titles_df = read_data_file("title.basics")
    crew_df = read_data_file("title.crew")
    ratings_df = read_data_file("title.ratings")
    directors_movies_df = directors_df.merge(
        crew_df, left_on="nconst", right_on="directors", how="inner"
    ).merge(titles_df, on="tconst", how="inner")[
        [
            "nconst",
            "tconst",
            "titleType",
            "primaryTitle",
            "originalTitle",
            "startYear",
            "runtimeMinutes",
            "genres",
        ]
    ]
    directors_movies_df = directors_movies_df.loc[
        directors_movies_df["titleType"] == "movie"
    ].merge(ratings_df, on="tconst", how="inner")
    directors_movies_df["isTopMovie"] = directors_movies_df["tconst"].isin(
        directors_top_movies_df["tconst"]
    )
    save_temp_file(directors_movies_df, file_name)
    return directors_movies_df


def format_num_votes(votes):
    if votes < 1000:
        return f"{votes/1000:.1f} K"
    elif votes >= 1000 and votes < 1e6:
        return f"{votes/1000:.0f} K"
    else:
        return f"{votes/1e6:.1f} M"


def modify_director_titles(director_titles: pd.DataFrame) -> pd.DataFrame:
    """
    This function modifies the 'director_titles' DataFrame by formatting and merging certain columns, creating a new column, and reordering the columns.

    Parameters:
        director_titles (pd.DataFrame): A pandas DataFrame containing director titles data.

    Returns:
        modified_df (pd.DataFrame): A modified DataFrame with formatted and merged columns.

    The function first formats the 'numVotes' column by converting values greater than or equal to 1000 into kilo format.
    Then, it merges the 'averageRating' and 'numVotes' columns into a single 'averageRating' column.
    Next, it merges the 'primaryTitle' and 'originalTitle' columns into a single 'primaryTitle' column, adding the original title in parentheses if it differs from the primary title.
    After that, it creates a new 'title' column with a hyperlink to the IMDb page of the movie.
    It then formats the 'runtimeMinutes' column into a 'total_minutes_formatted' column, converting total minutes into hours and minutes format.
    Finally, it reorders the columns and renames them before returning the modified DataFrame.
    """
    modified_df = director_titles.copy()

    modified_df[" "] = range(1, len(modified_df) + 1)

    modified_df["numVotes"] = modified_df["numVotes"].apply(format_num_votes)
    modified_df["isTopMovie"] = modified_df["isTopMovie"].map({True: "★", False: ""})
    # modified_df["averageRating"] = modified_df["averageRating"].astype(str)

    # Merge primaryTitle and originalTitle columns
    modified_df["primaryTitle"] = modified_df.apply(
        lambda row: row["primaryTitle"] + "<br>(" + row["originalTitle"] + ")"
        if row["originalTitle"] != row["primaryTitle"]
        else row["primaryTitle"],
        axis=1,
    )

    # Create a link column for the title
    modified_df["title"] = modified_df.apply(
        lambda row: f"[{row['primaryTitle']}](https://www.imdb.com/title/{row['tconst']})",
        axis=1,
    )

    def format_minutes(total_minutes):
        if total_minutes is None:
            return "-"
        total_minutes = int(total_minutes)
        hours = total_minutes // 60
        minutes = total_minutes % 60
        return f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"

    modified_df["total_minutes_formatted"] = modified_df["runtimeMinutes"].apply(
        format_minutes
    )
    modified_df["genres"] = modified_df["genres"].str.replace(",", ", ")
    # Reorder the columns
    modified = modified_df[
        [
            " ",
            "startYear",
            "isTopMovie",
            "averageRating",
            "numVotes",
            "title",
            "total_minutes_formatted",
            "genres",
        ]
    ].copy()
    modified.rename(
        columns={
            "startYear": "Year",
            "isTopMovie": "★",
            "averageRating": "Rate",
            "numVotes": "Votes",
            "title": "Title",
            "total_minutes_formatted": "Duration",
            "genres": "Genres",
        },
        inplace=True,
    )

    return modified


def get_genres(df: pd.DataFrame) -> str:
    # Join all row values to form a single string
    genres_string = ",".join(df["genres"])

    # Split the string by comma
    genres_list = genres_string.split(",")

    # Count the repetition of each value
    genre_counts = {}
    for genre in genres_list:
        if genre in genre_counts:
            genre_counts[genre] += 1
        else:
            genre_counts[genre] = 1

    # Sort the genres by count
    sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)

    # Keep only the top 8 genres
    top_genres = sorted_genres[:8]

    # Calculate repetition percent for each genre
    total_rows = len(df)
    genre_percentages = [
        (genre, count / total_rows * 100) for genre, count in top_genres
    ]

    # Form a comma-separated string containing values and their percent
    result_string = ", ".join(
        [f"{genre} ({percent:.0f}%)" for genre, percent in genre_percentages]
    )

    return result_string


def get_director_section_details(
    director_rank: int,
    director_name: str,
    director_id: str,
    birth_year: str,
    death_year: str,
    director_titles: pd.DataFrame,
) -> list[str]:
    markdown_content = []
    # Add a numbered section for the director
    lifespan = f" ({birth_year}-{death_year})" if death_year is not None else ""
    markdown_content.append(
        f'## {director_rank}. [{director_name}{lifespan}](https://www.imdb.com/name/{director_id}) <a name="dir{director_rank}"></a>'
    )
    markdown_content.append("### • Overview  ")

    def get_info(df: pd.DataFrame):
        first_year = df["startYear"].min()
        last_year = df["startYear"].max()
        num_titles = len(df)
        total_votes = df["numVotes"].sum()
        mean_rating = df["averageRating"].mean()
        genres = get_genres(df)
        return first_year, last_year, num_titles, total_votes, mean_rating, genres

    first_year, last_year, num_titles, total_votes, mean_rating, genres = get_info(
        director_titles
    )
    top_movies = director_titles[director_titles["isTopMovie"] == True]
    (
        top_first_year,
        top_last_year,
        top_num_titles,
        top_total_votes,
        top_mean_rating,
        top_genres,
    ) = get_info(top_movies)

    data = [
        (
            "Active&nbsp;Years",
            f"{first_year}-{last_year}",
            f"{top_first_year}-{top_last_year}",
        ),
        ("Num.&nbsp;Movies", num_titles, top_num_titles),
        (
            "Total&nbsp;Votes",
            format_num_votes(total_votes),
            format_num_votes(top_total_votes),
        ),
        (
            "Mean&nbsp;Rating",
            f"{mean_rating:.1f}",
            f"**<ins>{top_mean_rating:.2f}</ins>**",
        ),
        ("Genres", genres, top_genres),
    ]
    data = [(f"**{x[0]}**", x[1], x[2]) for x in data]
    markdown_content.append(
        tabulate(data, headers=("", "All Movies", "Top Movies (★)"), tablefmt="pipe")
    )
    # print(markdown_table)
    markdown_content.append("")
    return markdown_content


def get_director_section_titles(
    director_titles: pd.DataFrame, num_titles: int
) -> list[str]:
    markdown_content = []
    director_titles = director_titles.sort_values(
        by="averageRating", ascending=False
    ).head(num_titles)

    # Sort director_titles by startYear
    director_titles = director_titles.sort_values(by="startYear")

    director_titles = modify_director_titles(director_titles)
    markdown_content.append("### • Works  ")
    # Add the table of director's titles
    markdown_content.append(director_titles.to_markdown(index=False))
    markdown_content.append("")
    return markdown_content


def get_director_section(
    director_rank: int,
    director_name: str,
    director_id: str,
    birth_year: str,
    death_year: str,
    director_titles: pd.DataFrame,
    num_titles: int,
) -> list[str]:
    return get_director_section_details(
        director_rank,
        director_name,
        director_id,
        birth_year,
        death_year,
        director_titles,
    ) + get_director_section_titles(director_titles, num_titles)


def generate_report(num_directors, num_titles):
    directors_df, directors_top_movies_df = get_high_voted_directors()
    directors_all_movies_df = get_directors_all_movies(
        directors_df, directors_top_movies_df
    )

    # Sort directors_df by meanAverageRating in descending order
    directors_df = directors_df.sort_values(
        by=["meanAverageRating", "sumVotes"], ascending=False
    )

    # Select numDirectors directors with highest meanAverageRating
    directors_df = directors_df.head(num_directors)

    # Create an empty list to store the markdown content
    markdown_content = []
    table_of_contents = []
    # Loop over each director in directors_df
    for index, row in directors_df.iterrows():
        director_id = row["nconst"]
        director_name = row["director"]
        mean_rating = row["meanAverageRating"]
        birth_year = row["birthYear"]
        death_year = row["deathYear"]
        director_rank = index + 1
        table_of_contents.append(
            f"[{index+1}. {director_name} ({mean_rating:.2f})](#dir{director_rank})  "
        )
        # Filter selected_titles_df for the current director
        director_titles = directors_all_movies_df[
            directors_all_movies_df["nconst"] == director_id
        ]
        markdown_content += get_director_section(
            director_rank,
            director_name,
            director_id,
            birth_year,
            death_year,
            director_titles,
            num_titles,
        )

    markdown_content = table_of_contents + markdown_content

    # Join the markdown content list into a single string
    markdown_file_content = "\n".join(markdown_content)

    # Write the markdown content to a file
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(markdown_file_content)


temp_folder_path = crete_temp_folder()

generate_report(20, 15)
print("done")
