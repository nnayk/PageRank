import os
import random
import re
import sys
from collections import defaultdict

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(link for link in pages[filename] if link in pages)

    return pages


def transition_model(corpus, page, damping_factor) -> dict:
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.

    Params:
        corpus: Dict[page, links].
            - page: str
            - links: set(str)
                - set of all pages linked to by `page`
        page: str
            - current page
        damping_factor: float
    """
    random_factor = 1 - damping_factor
    distribution = {}
    num_outgoing = len(corpus[page])
    outgoing_probability = (
        0  # probability of choosing a single page that's linked to by page
    )
    random_probability = 1 / (
        len(corpus)
    )  # probability of selecting a single page from the corpus

    if num_outgoing > 0:
        random_probability = random_probability * random_factor

    # if there are outgoing links, set the associated probability
    if num_outgoing > 0:
        outgoing_probability = damping_factor * (1 / num_outgoing)
    for pg in corpus:
        distribution[pg] = random_probability
        if pg in corpus[page]:
            distribution[pg] += outgoing_probability
    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pg_ranks = defaultdict(
        lambda: 0
    )  # dictionary that maps pages to their page ranks
    curr_pg = random.choice(list(corpus.keys()))
    # sample n pages and store the number of occurrences of each page in the sample
    for _ in range(n):
        distribution = transition_model(corpus, curr_pg, damping_factor)
        curr_pg = random.choices(
            list(distribution.keys()), weights=list(distribution.values())
        )[0]
        pg_ranks[curr_pg] = pg_ranks[curr_pg] + 1
    # calculate the page rank based on number of samples
    for pg in pg_ranks:
        pg_ranks[pg] /= n
    return pg_ranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    done = False
    num_pgs = len(corpus)
    pg_ranks = {key: 1 / num_pgs for key in corpus.keys()}
    random_factor = 1 - damping_factor
    random_probability = random_factor / num_pgs
    # store the number of outgoing links for each pg
    outgoing_counts = {}
    for pg, links in corpus.items():
        outgoing_counts[pg] = len(links)
    # store the set of links that point to each page
    source_links = defaultdict(lambda: set())
    for pg in corpus:
        for dest in corpus[pg]:
            source_links[dest].add(pg)
    while not done:
        done = True
        for pg in corpus:
            curr_rank = pg_ranks[pg]
            new_rank = random_probability
            raw_source_sum = 0
            for source in source_links[pg]:
                raw_source_sum += (pg_ranks[source]) / (outgoing_counts[source])
            new_rank += damping_factor * raw_source_sum
            if abs(new_rank - curr_rank) > 0.001:
                done = False
            pg_ranks[pg] = new_rank
    return pg_ranks


if __name__ == "__main__":
    main()
