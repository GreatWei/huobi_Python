class RestApiRequest(object):

    def __init__(self):
        self.method = ""
        self.url = ""
        self.host = ""
        self.post_body = ""
        self.header = dict()
        self.json_parser = None
        self.proxies = {"http": "127.0.0.1:10809", "https": "127.0.0.1:10809"}
