"""add_schema_to_system_prompt must embed the output schema as JSON, not as a Python dict repr.

ChatOpenAI and ChatAzureOpenAI (Responses API) interpolated the schema dict straight into the
f-string, so the model saw single quotes and True/False inside <json_schema>. The other
providers already use json.dumps(). This option is meant for endpoints that ignore
response_format, where the prompt is the only place the model sees the schema.
"""

import json
from types import SimpleNamespace

from pydantic import BaseModel

from browser_use.llm.azure.chat import ChatAzureOpenAI
from browser_use.llm.messages import SystemMessage, UserMessage
from browser_use.llm.openai.chat import ChatOpenAI


class Answer(BaseModel):
	answer: str
	done: bool


MESSAGES = [SystemMessage(content='You are a helpful agent.'), UserMessage(content='Say hi')]


def _schema_from_system_prompt(content: str) -> dict:
	assert '<json_schema>' in content
	schema_text = content.split('<json_schema>', 1)[1].split('</json_schema>', 1)[0]
	return json.loads(schema_text)


async def test_openai_schema_in_system_prompt_is_valid_json(monkeypatch):
	captured: dict = {}

	class FakeCompletions:
		async def create(self, **kwargs):
			captured.update(kwargs)
			return SimpleNamespace(
				choices=[
					SimpleNamespace(message=SimpleNamespace(content='{"answer": "hi", "done": true}'), finish_reason='stop')
				],
				usage=None,
			)

	llm = ChatOpenAI(model='gpt-4.1-mini', api_key='test-key', add_schema_to_system_prompt=True)
	monkeypatch.setattr(llm, 'get_client', lambda: SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions())))

	result = await llm.ainvoke(MESSAGES, output_format=Answer)

	assert result.completion == Answer(answer='hi', done=True)
	schema = _schema_from_system_prompt(captured['messages'][0]['content'])
	assert schema == captured['response_format']['json_schema']['schema']
	assert schema['properties']['done'] == {'type': 'boolean'}


async def test_azure_responses_schema_in_system_prompt_is_valid_json(monkeypatch):
	captured: dict = {}

	class FakeResponses:
		async def create(self, **kwargs):
			captured.update(kwargs)
			return SimpleNamespace(output_text='{"answer": "hi", "done": true}', usage=None, status='completed')

	llm = ChatAzureOpenAI(
		model='gpt-4.1-mini',
		api_key='test-key',
		azure_endpoint='https://example.openai.azure.com',
		use_responses_api=True,
		add_schema_to_system_prompt=True,
	)
	monkeypatch.setattr(llm, 'get_client', lambda: SimpleNamespace(responses=FakeResponses()))

	result = await llm.ainvoke(MESSAGES, output_format=Answer)

	assert result.completion == Answer(answer='hi', done=True)
	schema = _schema_from_system_prompt(captured['input'][0]['content'])
	assert schema == captured['text']['format']['schema']
	assert schema['properties']['done'] == {'type': 'boolean'}
