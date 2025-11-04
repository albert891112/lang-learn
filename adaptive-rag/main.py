from graph.service import RAG_GRAPH


async def fact_rag():
    config = {"recursion_limit": 50}
    inputs = {
        "question": "網路上傳言非洲豬瘟的防疫預算沒有被刪除，請問是真的嗎？",
        "max_retries": 3,
    }
    async for event in RAG_GRAPH.astream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)


if __name__ == "__main__":
    import asyncio

    asyncio.run(fact_rag())
