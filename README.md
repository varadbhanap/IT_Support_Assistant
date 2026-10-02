# Internal IT Support Assistant

A conversational assistant built with LangChain that answers employee IT questions by combining document retrieval, tool use, and conversation memory.

## What it does

- Answers questions about internal IT policies (remote work, VPN access, onboarding, device security) by searching a small set of sample policy documents
- Performs basic arithmetic calculations
- Looks up the status of a support ticket by ID
- Decides on its own which capability to use based on the question
- Remembers earlier turns in the same conversation

## How it works

Policy documents are split into chunks using `RecursiveCharacterTextSplitter`, then embedded using a custom TF-IDF based embeddings class and stored in a FAISS vector store. The retriever is exposed to the agent as a tool alongside a calculator and a ticket lookup tool. The agent itself is built with `create_agent`, LangChain's current (v1.0+) agent constructor, backed by Google's Gemini model, with conversation memory handled through a LangGraph checkpointer.

## Notes

- The embeddings class is a lightweight, fully offline TF-IDF implementation, not a production grade embedding model. It can be swapped for a real embedding model without changing anything else in the pipeline, since every downstream component only depends on the `Embeddings` interface.
- Tickets and policy documents are hardcoded sample data for demonstration. A production version would connect to real data sources.
