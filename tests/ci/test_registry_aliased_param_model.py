"""Regression test: custom actions whose param_model uses field aliases.

The action JSON schema the LLM sees uses the aliases, but `Tools.act()` dumps the validated
action by field name before `Registry.execute_action()` re-validates it, and the sensitive-data
pass does the same round-trip. A param model like `Field(alias='userName')` therefore failed
with "Field required" on every call, so aliased custom actions could never run.
"""

from pydantic import BaseModel, Field

from browser_use.agent.views import ActionResult
from browser_use.tools.registry.service import Registry
from browser_use.tools.service import Tools


class LoginParams(BaseModel):
	user_name: str = Field(alias='userName')
	remember_me: bool = Field(default=False, alias='rememberMe')


def _tools_with_login_action() -> Tools:
	tools = Tools()

	@tools.action('Log in with a user name', param_model=LoginParams)
	async def login(params: LoginParams):
		return ActionResult(extracted_content=f'{params.user_name} remember={params.remember_me}')

	return tools


async def test_aliased_param_model_runs_through_act():
	tools = _tools_with_login_action()
	ActionModel = tools.registry.create_action_model()

	# The LLM fills the schema, which uses the aliases.
	action = ActionModel.model_validate({'login': {'userName': 'bob', 'rememberMe': True}})
	result = await tools.act(action, browser_session=None)  # type: ignore[arg-type]

	assert result.error is None
	assert result.extracted_content == 'bob remember=True'


async def test_aliased_param_model_accepts_alias_and_field_names():
	registry = Registry()

	@registry.action('Log in with a user name', param_model=LoginParams)
	async def login(params: LoginParams):
		return ActionResult(extracted_content=params.user_name)

	by_alias = await registry.execute_action('login', {'userName': 'bob'})
	by_name = await registry.execute_action('login', {'user_name': 'alice'})

	assert by_alias.extracted_content == 'bob'
	assert by_name.extracted_content == 'alice'


async def test_aliased_param_model_with_sensitive_data():
	registry = Registry()

	@registry.action('Log in with a user name', param_model=LoginParams)
	async def login(params: LoginParams):
		return ActionResult(extracted_content=params.user_name)

	result = await registry.execute_action(
		'login',
		{'userName': '<secret>login_user</secret>'},
		sensitive_data={'login_user': 'carol'},
	)

	assert result.extracted_content == 'carol'
