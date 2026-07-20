"""Render mixins — the impl half of the contract render-mixin models.

:class:`tai_contract.template.ConditionMixin` / ``ExprMixin`` carry the field
shape (the contract); these subclasses add the ``rendered_*`` methods that reach
the live ``resource_manager`` to render those fields.
"""

from __future__ import annotations

from tai_contract.app import tai_app
from tai_contract.template import ConditionMixin as ConditionFields
from tai_contract.template import ExprMixin as ExprFields


class ConditionMixin(ConditionFields):
    async def rendered_condition(self) -> str:
        return await tai_app.storage.resource_manager.render_by_id_or_content(
            content=self.condition,
            template_id=self.condition_id,
            kwargs=self.condition_kwargs,
        )


class ExprMixin(ExprFields):
    async def rendered_expr(self) -> str:
        return await tai_app.storage.resource_manager.render_by_id_or_content(
            content=self.expr,
            template_id=self.expr_id,
            kwargs=self.expr_kwargs,
        )


__all__ = ["ConditionMixin", "ExprMixin"]
