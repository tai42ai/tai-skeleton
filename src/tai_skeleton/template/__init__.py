"""Resource feature: the load/render impl over a :class:`~tai_contract.storage.Storage`.

:class:`ResourceManager` loads content by storage ``id``, ``url``, or a raw file —
as text OR media — and renders Jinja templates fetched through the active
``Storage`` provider; the render mixins add ``rendered_*`` methods to the contract
field models; ``TemplateCacheSettings`` configures the compiled-template cache.
``MediaBlock`` / ``ContentPart`` are the media types it produces.
"""

from tai_skeleton.template.media import ContentPart, MediaBlock
from tai_skeleton.template.mixins import ConditionMixin, ExprMixin
from tai_skeleton.template.resource_manager import ResourceManager, TemplateNotFoundError
from tai_skeleton.template.settings import TemplateCacheSettings, template_cache_settings

__all__ = [
    "ConditionMixin",
    "ContentPart",
    "ExprMixin",
    "MediaBlock",
    "ResourceManager",
    "TemplateCacheSettings",
    "TemplateNotFoundError",
    "template_cache_settings",
]
