from assistant.memory.file_provider import FileMemoryProvider


async def test_write_search_roundtrip(tmp_path):
    m = FileMemoryProvider(tmp_path)
    await m.write("User prefers tabs over spaces", tags=["preference"])
    await m.write("Project uses Python 3.12", tags=["project"])
    hits = await m.search("tabs")
    assert len(hits) == 1 and "tabs" in hits[0]["content"]


async def test_search_ranks_by_overlap(tmp_path):
    m = FileMemoryProvider(tmp_path)
    await m.write("python testing with pytest")
    await m.write("python is a language")
    hits = await m.search("python pytest testing")
    assert hits[0]["content"] == "python testing with pytest"


async def test_tags_are_searchable(tmp_path):
    m = FileMemoryProvider(tmp_path)
    await m.write("some note", tags=["deployment"])
    hits = await m.search("deployment")
    assert len(hits) == 1


async def test_persistence_across_instances(tmp_path):
    await FileMemoryProvider(tmp_path).write("remember me")
    later = await FileMemoryProvider(tmp_path).all()
    assert any(e["content"] == "remember me" for e in later)


async def test_prefetch_formats_block(tmp_path):
    m = FileMemoryProvider(tmp_path)
    await m.write("alpha fact")
    assert await m.prefetch("alpha") == "- alpha fact"


async def test_empty_query_returns_nothing(tmp_path):
    m = FileMemoryProvider(tmp_path)
    await m.write("x")
    assert await m.search("") == []
