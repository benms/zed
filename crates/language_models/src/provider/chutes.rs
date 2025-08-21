use anyhow::{Context as _, Result, anyhow};
use credentials_provider::CredentialsProvider;

use futures::Stream;
use futures::{FutureExt, StreamExt, future::BoxFuture};
use gpui::{AnyView, App, AsyncApp, Context, Entity, Subscription, Task, Window};
use http_client::HttpClient;
use language_model::{
    AuthenticateError, LanguageModel, LanguageModelCompletionEvent, LanguageModelCompletionError,
    LanguageModelId, LanguageModelName, LanguageModelProvider, LanguageModelProviderId,
    LanguageModelProviderName, LanguageModelProviderState, LanguageModelRequest,
    MessageContent, RateLimiter, Role, StopReason, TokenUsage,
};
use chutes::{Model, stream_completion};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use settings::{Settings, SettingsStore};
use std::pin::Pin;
use std::sync::Arc;
use strum::IntoEnumIterator;

use ui::{List, ListItem, ListSeparator, prelude::*};
use ui_input::SingleLineInput;
use util::ResultExt;

use crate::{AllLanguageModelSettings, ui::InstructionListItem};

const PROVIDER_ID: LanguageModelProviderId = LanguageModelProviderId::new("chutes");
const PROVIDER_NAME: LanguageModelProviderName = LanguageModelProviderName::new("Chutes.ai");

#[derive(Default, Clone, Debug, PartialEq)]
pub struct ChutesSettings {
    pub api_url: String,
    pub available_models: Vec<AvailableModel>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AvailableModel {
    pub name: String,
    pub display_name: Option<String>,
    pub max_tokens: u64,
    pub max_output_tokens: Option<u64>,
}

pub struct ChutesLanguageModelProvider {
    http_client: Arc<dyn HttpClient>,
    state: gpui::Entity<State>,
}

pub struct State {
    api_key: Option<String>,
    api_key_from_env: bool,
    _subscription: Subscription,
}

const CHUTES_API_KEY_VAR: &str = "CHUTES_API_KEY";

impl State {
    fn is_authenticated(&self) -> bool {
        self.api_key.is_some()
    }

    fn reset_api_key(&self, cx: &mut Context<Self>) -> Task<Result<()>> {
        let credentials_provider = <dyn CredentialsProvider>::global(cx);
        let api_url = AllLanguageModelSettings::get_global(cx)
            .chutes
            .as_ref()
            .map(|settings| settings.api_url.clone())
            .unwrap_or_else(|| chutes::CHUTES_API_URL.to_string());
        cx.spawn(async move |this, cx| {
            credentials_provider
                .delete_credentials(&api_url, cx)
                .await
                .log_err();
            this.update(cx, |this, cx| {
                this.api_key = None;
                this.api_key_from_env = false;
                cx.notify();
            })
        })
    }

    fn set_api_key(&mut self, api_key: String, cx: &mut Context<Self>) -> Task<Result<()>> {
        let credentials_provider = <dyn CredentialsProvider>::global(cx);
        let api_url = AllLanguageModelSettings::get_global(cx)
            .chutes
            .as_ref()
            .map(|settings| settings.api_url.clone())
            .unwrap_or_else(|| chutes::CHUTES_API_URL.to_string());
        cx.spawn(async move |this, cx| {
            credentials_provider
                .write_credentials(&api_url, "Bearer", api_key.as_bytes(), cx)
                .await
                .log_err();
            this.update(cx, |this, cx| {
                this.api_key = Some(api_key);
                cx.notify();
            })
        })
    }

    fn authenticate(&self, cx: &mut Context<Self>) -> Task<Result<(), AuthenticateError>> {
        if self.is_authenticated() {
            return Task::ready(Ok(()));
        }

        let credentials_provider = <dyn CredentialsProvider>::global(cx);
        let api_url = AllLanguageModelSettings::get_global(cx)
            .chutes
            .as_ref()
            .map(|settings| settings.api_url.clone())
            .unwrap_or_else(|| chutes::CHUTES_API_URL.to_string());
        cx.spawn(async move |this, cx| {
            let (api_key, from_env) = if let Ok(api_key) = std::env::var(CHUTES_API_KEY_VAR) {
                (api_key, true)
            } else {
                let (_, api_key) = credentials_provider
                    .read_credentials(&api_url, cx)
                    .await?
                    .ok_or(AuthenticateError::CredentialsNotFound)?;
                (
                    String::from_utf8(api_key).context("invalid Chutes.ai API key")?,
                    false,
                )
            };
            this.update(cx, |this, cx| {
                this.api_key = Some(api_key);
                this.api_key_from_env = from_env;
                cx.notify();
            })?;

            Ok(())
        })
    }
}

impl ChutesLanguageModelProvider {
    pub fn new(http_client: Arc<dyn HttpClient>, cx: &mut App) -> Self {
        let state = cx.new(|cx| State {
            api_key: None,
            api_key_from_env: false,
            _subscription: cx.observe_global::<SettingsStore>(|_this: &mut State, cx| {
                cx.notify();
            }),
        });

        Self { http_client, state }
    }
}

impl LanguageModelProvider for ChutesLanguageModelProvider {
    fn id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn icon(&self) -> ui::IconName {
        ui::IconName::ZedAssistant
    }

    fn provided_models(&self, cx: &App) -> Vec<Arc<dyn LanguageModel>> {
        let mut models = Vec::new();

        // Add built-in models
        for model in Model::iter() {
            if let Model::Custom { .. } = model {
                continue;
            }

            models.push(Arc::new(ChutesLanguageModel {
                id: LanguageModelId::from(format!("chutes::{}", model.id())),
                model,
                http_client: self.http_client.clone(),
                request_limiter: RateLimiter::new(4),
                state: self.state.clone(),
            }) as Arc<dyn LanguageModel>);
        }

        // Add custom models from settings
        if let Some(chutes_settings) = AllLanguageModelSettings::get_global(cx).chutes.as_ref() {
            for model in &chutes_settings.available_models {
                models.push(Arc::new(ChutesLanguageModel {
                    id: LanguageModelId::from(format!("chutes::{}", model.name)),
                    model: Model::Custom {
                        name: model.name.clone(),
                        display_name: model.display_name.clone(),
                        max_tokens: model.max_tokens,
                        max_output_tokens: model.max_output_tokens,
                    },
                    http_client: self.http_client.clone(),
                    request_limiter: RateLimiter::new(4),
                    state: self.state.clone(),
                }) as Arc<dyn LanguageModel>);
            }
        }

        models
    }

    fn is_authenticated(&self, cx: &App) -> bool {
        self.state.read(cx).is_authenticated()
    }

    fn authenticate(&self, cx: &mut App) -> Task<Result<(), AuthenticateError>> {
        self.state.update(cx, |state, cx| state.authenticate(cx))
    }

    fn configuration_view(&self, _agent: language_model::ConfigurationViewTargetAgent, window: &mut Window, cx: &mut App) -> AnyView {
        cx.new(|cx| ConfigurationView::new(self.state.clone(), window, cx))
            .into()
    }

    fn reset_credentials(&self, cx: &mut App) -> Task<anyhow::Result<()>> {
        self.state.update(cx, |state, cx| state.reset_api_key(cx))
    }

    fn default_model(&self, cx: &App) -> Option<Arc<dyn LanguageModel>> {
        self.provided_models(cx)
            .into_iter()
            .find(|model| model.name().0.contains("llama-3.1-405b"))
    }

    fn default_fast_model(&self, cx: &App) -> Option<Arc<dyn LanguageModel>> {
        self.provided_models(cx)
            .into_iter()
            .find(|model| model.name().0.contains("llama-3.1-70b"))
            .or_else(|| self.provided_models(cx).into_iter().next())
    }

}

impl LanguageModelProviderState for ChutesLanguageModelProvider {
    type ObservableEntity = State;

    fn observable_entity(&self) -> Option<gpui::Entity<Self::ObservableEntity>> {
        Some(self.state.clone())
    }
}

pub struct ChutesLanguageModel {
    id: LanguageModelId,
    model: Model,
    http_client: Arc<dyn HttpClient>,
    request_limiter: RateLimiter,
    state: Entity<State>,
}

impl ChutesLanguageModel {
    fn to_chutes_message(message: &language_model::LanguageModelRequestMessage) -> chutes::Message {
        let mut content_text = String::new();
        
        for content in &message.content {
            match content {
                MessageContent::Text(text) => {
                    if !content_text.is_empty() {
                        content_text.push('\n');
                    }
                    content_text.push_str(text);
                }
                MessageContent::Image(_image_data) => {
                    if !content_text.is_empty() {
                        content_text.push('\n');
                    }
                    content_text.push_str("[Image content]");
                }
                _ => {
                    // Handle other content types (thinking, tool use, etc.)
                    if let Some(str_content) = content.to_str() {
                        if !content_text.is_empty() {
                            content_text.push('\n');
                        }
                        content_text.push_str(str_content);
                    }
                }
            }
        }
        
        chutes::Message {
            role: match message.role {
                Role::User => chutes::Role::User,
                Role::Assistant => chutes::Role::Assistant,
                Role::System => chutes::Role::System,
            },
            content: content_text,
        }
    }
}

impl LanguageModel for ChutesLanguageModel {
    fn id(&self) -> LanguageModelId {
        self.id.clone()
    }

    fn name(&self) -> LanguageModelName {
        LanguageModelName::from(self.model.display_name().to_string())
    }

    fn provider_id(&self) -> LanguageModelProviderId {
        PROVIDER_ID
    }

    fn provider_name(&self) -> LanguageModelProviderName {
        PROVIDER_NAME
    }

    fn telemetry_id(&self) -> String {
        format!("chutes::{}", self.model.id())
    }

    fn max_token_count(&self) -> u64 {
        self.model.max_token_count()
    }

    fn count_tokens(
        &self,
        request: LanguageModelRequest,
        _cx: &App,
    ) -> BoxFuture<'static, Result<u64>> {
        // Simple token estimation - in a real implementation you'd use tiktoken or similar
        let text = request
            .messages
            .iter()
            .map(|msg| {
                msg.content.iter().map(|content| {
                    match content {
                        MessageContent::Text(text) => text.len(),
                        MessageContent::Image(_) => 100, // Estimate for image
                        _ => content.to_str().map_or(0, |s| s.len()),
                    }
                }).sum::<usize>()
            })
            .sum::<usize>() as u64;
        
        async move { Ok(text / 4) }.boxed() // Rough estimate: 4 chars per token
    }

    fn stream_completion(
        &self,
        request: LanguageModelRequest,
        cx: &AsyncApp,
    ) -> BoxFuture<'static, Result<Pin<Box<dyn Stream<Item = Result<LanguageModelCompletionEvent, LanguageModelCompletionError>> + Send + 'static>>, LanguageModelCompletionError>> {
        let http_client = self.http_client.clone();
        let state = self.state.clone();
        let model = self.model.clone();
        let request_limiter = self.request_limiter.clone();

        let api_key_result = state.read_with(cx, |state, _cx| state.api_key.clone());
        
        async move {
            let api_key = api_key_result
                .map_err(|e| LanguageModelCompletionError::Other(e))?
                .ok_or_else(|| LanguageModelCompletionError::Other(anyhow!("missing API key")))?;

            let chutes_request = chutes::Request {
                model: model.id().to_string(),
                messages: request
                    .messages
                    .iter()
                    .map(Self::to_chutes_message)
                    .collect(),
                max_tokens: Some(4096), // Default max tokens since field doesn't exist
                temperature: request.temperature,
                stream: true,
            };

            let future = request_limiter.stream(async move {
                let stream = stream_completion(
                    http_client.as_ref(),
                    chutes::CHUTES_API_URL,
                    &api_key,
                    chutes_request,
                ).await?;

                Ok(stream
                    .map(|response| {
                        match response {
                            Ok(response) => {
                                if let Some(choice) = response.choices.first() {
                                    if let Some(delta) = &choice.delta {
                                        if let Some(content) = &delta.content {
                                            return Ok(LanguageModelCompletionEvent::Text(content.clone()));
                                        }
                                    }
                                    if let Some(finish_reason) = &choice.finish_reason {
                                        return Ok(LanguageModelCompletionEvent::Stop(
                                            match finish_reason.as_str() {
                                                "stop" => StopReason::EndTurn,
                                                "length" => StopReason::MaxTokens,
                                                _ => StopReason::EndTurn,
                                            }
                                        ));
                                    }
                                }
                                if let Some(usage) = response.usage {
                                    return Ok(LanguageModelCompletionEvent::UsageUpdate(TokenUsage {
                                        input_tokens: usage.prompt_tokens,
                                        output_tokens: usage.completion_tokens,
                                        cache_creation_input_tokens: 0,
                                        cache_read_input_tokens: 0,
                                    }));
                                }
                                Ok(LanguageModelCompletionEvent::Text(String::new()))
                            }
                            Err(error) => Err(LanguageModelCompletionError::Other(anyhow!(error))),
                        }
                    })
                    .boxed())
            });
            Ok(future.await?.boxed())
        }.boxed()
    }

    fn supports_tools(&self) -> bool {
        false
    }

    fn supports_images(&self) -> bool {
        true
    }

    fn supports_tool_choice(&self, _tool_choice: language_model::LanguageModelToolChoice) -> bool {
        false
    }
}

struct ConfigurationView {
    api_key_editor: Entity<SingleLineInput>,
    state: Entity<State>,
}

impl ConfigurationView {
    fn new(state: Entity<State>, window: &mut Window, cx: &mut Context<Self>) -> Self {
        let api_key_editor = cx.new(|cx| {
            SingleLineInput::new(
                window,
                cx,
                "Enter your Chutes.ai API key",
            )
        });

        Self {
            api_key_editor,
            state,
        }
    }

    fn save_api_key(&mut self, cx: &mut Context<Self>) {
        let api_key = self.api_key_editor.read(cx).text(cx);
        if !api_key.trim().is_empty() {
            self.state.update(cx, |state, cx| {
                state.set_api_key(api_key.trim().to_string(), cx).detach_and_log_err(cx);
            });
        }
    }
}

impl ui::Render for ConfigurationView {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        const INSTRUCTIONS: [&str; 2] = [
            "To use Chutes.ai models, you need to add your Chutes.ai API key.",
            "You can create an API key at: https://chutes.ai/dashboard",
        ];

        v_flex()
            .gap_2()
            .child(
                List::new()
                    .child(
                        v_flex()
                            .gap_2()
                            .child(Label::new("Instructions").size(LabelSize::Small))
                            .child(
                                v_flex()
                                    .gap_1()
                                    .children(INSTRUCTIONS.iter().map(|instruction| {
                                        InstructionListItem::new(*instruction, None::<String>, None::<String>)
                                    })),
                            ),
                    )
                    .child(ListSeparator)
                    .child(
                        ListItem::new("api_key")
                            .spacing(ui::ListItemSpacing::Sparse)
                            .child(Label::new("API Key").size(LabelSize::Small))
                            .child(
                                h_flex()
                                    .w_full()
                                    .justify_end()
                                    .child(self.api_key_editor.clone()),
                            ),
                    )
                    .child(ListSeparator)
                    .child(
                        ListItem::new("save_button")
                            .spacing(ui::ListItemSpacing::Sparse)
                            .child(
                                Button::new("save", "Save API Key")
                                    .style(ButtonStyle::Filled)
                                    .size(ButtonSize::Compact)
                                    .on_click(cx.listener(|this, _, _, cx| {
                                        this.save_api_key(cx);
                                    })),
                            ),
                    ),
            )
    }
}

