import itertools

from docutils import nodes
from myst_nb.core.render import MimeData, NbElementRenderer


class MimeRenderPlugin:
    """Protocol for a mime renderer plugin."""

    json_ids = itertools.count()

    @staticmethod
    def handle_mime(renderer: NbElementRenderer, data: MimeData, inline: bool):
        if data.mime_type == "application/json":
            json_id = next(MimeRenderPlugin.json_ids)
            html = (
                f"<pre id='json-{json_id}'></pre>"
                f"<script>document.getElementById('json-{json_id}').textContent = "
                f"JSON.stringify({data.string}, undefined, 2);</script>"
            )
            return [
                nodes.raw(text=html, format="html", classes=["output", "text_html"]),
            ]
        return None
