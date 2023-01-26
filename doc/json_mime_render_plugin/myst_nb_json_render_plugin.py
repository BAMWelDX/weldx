import itertools

from docutils import nodes
from IPython.core.display import JSON
from myst_nb.core.render import MimeData, NbElementRenderer
from sphinx.util.logging import getLogger

log = getLogger("sphinx")

JSON


class MimeRenderPlugin:
    """Protocol for a mime renderer plugin."""

    json_ids = itertools.count()

    @staticmethod
    def handle_mime(renderer: NbElementRenderer, data: MimeData, inline: bool):
        # TODO: what about inline?
        # if not inline and data.mime_type == "application/json":
        if data.mime_type == "application/json":
            json_id = next(MimeRenderPlugin.json_ids)
            html = (
                f"<pre id='json-{json_id}'></pre>"
                f"<script>document.getElementById('json-{json_id}').textContent = "
                f"JSON.stringify({data.string}, undefined, 2);</script>"
            )
            log.info("html: %s", html)
            return [
                nodes.raw(text=html, format="html", classes=["output", "text_html"]),
            ]
        return None
