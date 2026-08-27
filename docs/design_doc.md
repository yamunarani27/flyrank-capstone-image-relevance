# Design Doc — AI Image Understanding & Content Matching Engine

## Problem

Given a small library of images and a set of blog posts, automatically tag each image by what it actually depicts, and recommend the best-matching image for each post based on meaning — not filenames or keywords. Critically,the system must know when it doesn't have a good match and say so, rather than guess. Wrong recommendations are worse than no recommendation.

## Non-goal

This is not a general-purpose image search engine — it does not aim to scale beyond a small, bounded corpus (~50 images) or support arbitrary open-ended queries. No UI is built; the review workflow is API endpoints only. Comparing multiple vision or embedding models is out of scope — one of each is enough.

