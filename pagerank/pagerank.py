import os
import random
import re
import sys
import math

"""
Shuyan Liu
CS50's Intro to AI
07/04/2020
"""

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
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    probabilities = corpus.copy()
    totalPages = len(corpus)
    links = corpus[page]
    numLinks = len(links)
    # If there are no links on this page, all pages has equal probability
    if numLinks == 0:
        for i in probabilities:
            probabilities[i] = 1 / totalPages
    # If there are links...
    else:
        # "With probability 1 - damping_factor, the random surfer should
        # randomly choose one of all pages in the corpus with equal probability"
        for i in probabilities:
            probabilities[i] = (1 - damping_factor) / totalPages
        # "With probability damping_factor, the random surfer should
        # randomly choose one of the links from page with equal probability"
        for i in links:
            probabilities[i] += damping_factor / numLinks
    return probabilities


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pagerank = corpus.copy()
    for i in pagerank:
        pagerank[i] = 0
    # Choose the first page randomly
    page = random.choice(list(corpus.keys()))
    pagerank[page] += 1
    # Choose the following pages with probabilities defined by the transition model
    for i in range(n-1):
        probabilities = transition_model(corpus, page, damping_factor)
        weight = list(probabilities.values())
        page = random.choices(list(corpus.keys()), weights=weight, cum_weights=None, k=1)
        page = page[0]
        pagerank[page] += 1
    # Normalize the values so they add to 1
    for i in pagerank:
        pagerank[i] /= n
    return pagerank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pagerank = corpus.copy()
    totalPages = len(corpus)
    # Assign initial pageranks
    for i in pagerank:
        pagerank[i] = 1 / totalPages
    # If a page has no links, then it can be interpreted as having links to every page, including itself
    for page in corpus:
        if len(corpus[page]) == 0:
            corpus[page] = corpus.keys()
    newPagerank = pagerank.copy()
    isConvergent = False
    while not isConvergent:
        pagerank = newPagerank.copy()
        for page in corpus:
            links = set()
            # Add all pages that link to the current page
            for i in corpus:
                if page in corpus[i]:
                    links.add(i)
            # Calculation for the summation part of the formula
            count = 0
            for i in links:
                numLinks = len(corpus[i])
                count += pagerank[i] / numLinks
            # Assign new values with formula
            newPagerank[page] = ((1 - damping_factor) / totalPages) + (damping_factor * count)
        flag = True
        # If all values are within 0.001 of those of the previous pagerank, then it is convergent
        for page in corpus:
            if math.fabs(pagerank[page] - newPagerank[page]) > 0.001:
                flag = False
        if flag:
            isConvergent = True
    return pagerank


if __name__ == "__main__":
    main()
