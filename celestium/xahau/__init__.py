#!/usr/bin/env python
# coding: utf-8

from xahau.wallet import Wallet
from xahau.clients import WebsocketClient
from xahau.models import Transaction
from xahau.transaction import sign_and_submit


class XahauBot(object):
    _client: WebsocketClient = None

    def __init__(cls) -> None:
        cls._client = WebsocketClient("wss://xahau-test.net")
        pass

    def submit(cls, wallet: Wallet, tx: Transaction):
        try:
            with cls._client as _:
                return sign_and_submit(tx, cls._client, wallet, autofill=True)
        except Exception as e:
            print(f"Error: {e}")
            return None
