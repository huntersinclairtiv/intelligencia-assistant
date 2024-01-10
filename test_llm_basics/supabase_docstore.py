import os
from dotenv import load_dotenv
from typing import (
    Dict,
    Iterator,
    Optional,
    Sequence,
)

from supabase.client import Client, create_client
from langchain.docstore.document import Document
from langchain_core.stores import BaseStore


class SupabaseDocstore(BaseStore):
    """
    class for handling docstore operations on supabase
    """

    def __init__(self, table_name) -> None:
        load_dotenv()
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        self.supabase: Client = create_client(supabase_url, supabase_key)
        self.table_name = table_name

    def format_response(self, item_list) -> Dict:
        store = {}
        for item in item_list:
            store[item['id']] = Document(
                page_content=item['content'], metadata=item['metadata'])
        return store

    def mget(self, keys: Sequence[str]):
        # TODO: Add where clause with 'id IN (<id_list>)'.
        response = self.supabase.table(self.table_name).select("*").execute()
        doc_store = self.format_response(response.data)
        return [doc_store.get(key) for key in keys]

    def mset(self, key_value_pairs) -> None:
        doc_list = []
        for key, value in key_value_pairs:
            doc_list.append(
                {'id': key, 'content': value.page_content, 'metadata': value.metadata})
        self.supabase.table(self.table_name).upsert(doc_list).execute()

    def mdelete(self, keys: Sequence[str]) -> None:
        self.supabase.table(self.table_name).select(
            '*').in_('id', keys).execute()

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        response = self.supabase.table(self.table_name).select("*").execute()
        doc_store = self.format_response(response.data)
        if prefix is None:
            yield from doc_store.keys()
        else:
            for key in doc_store.keys():
                if key.startswith(prefix):
                    yield key
