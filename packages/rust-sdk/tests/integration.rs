use std::env;

use futures_util::StreamExt;
use kothagpt::types::{AgentSpec, ChatCompletionRequest, Message};
use kothagpt::{KothaGPT, WebSocketClient};

fn base_url() -> String {
    env::var("KOTHAGPT_API_URL").unwrap_or_else(|_| "http://localhost:8000".into())
}

fn ws_base_url() -> String {
    let url = base_url();
    if url.starts_with("https") {
        url.replace("https", "wss")
    } else {
        url.replace("http", "ws")
    }
}

#[tokio::test]
async fn models_list() {
    let client = KothaGPT::with_base_url(&base_url()).unwrap();
    let models = client.models().await.unwrap();
    assert!(!models.is_empty());
    assert_eq!(models[0].id, "kothagpt");
}

#[tokio::test]
async fn chat_create() {
    let client = KothaGPT::with_base_url(&base_url()).unwrap();
    let request = ChatCompletionRequest::new(vec![Message::user("হ্যালো")]);
    let completion = client.chat(request).await.unwrap();
    assert!(!completion.choices.is_empty());
    assert!(!completion.text().is_empty());
    assert!(completion.usage.total_tokens > 0);
}

#[tokio::test]
async fn chat_stream() {
    let client = KothaGPT::with_base_url(&base_url()).unwrap();
    let request = ChatCompletionRequest::new(vec![Message::user("হ্যালো")]);
    let mut stream = client.chat_stream(request).await.unwrap();
    let mut count = 0usize;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.unwrap();
        assert!(!chunk.choices.is_empty());
        count += 1;
    }
    assert!(count > 0);
}

#[tokio::test]
async fn embeddings() {
    let client = KothaGPT::with_base_url(&base_url()).unwrap();
    let response = client.embed(vec!["বাংলা".into(), "ভাষা".into()]).await.unwrap();
    assert_eq!(response.data.len(), 2);
    assert_eq!(response.data[0].embedding.len(), 256);
}

#[tokio::test]
async fn rerank() {
    let client = KothaGPT::with_base_url(&base_url()).unwrap();
    let response = client
        .rerank("বাংলা ভাষা", vec!["অন্য কিছু".into(), "বাংলা ভাষা শেখা".into()])
        .await
        .unwrap();
    assert_eq!(response.results.len(), 2);
}

#[tokio::test]
async fn tools() {
    let client = KothaGPT::with_base_url(&base_url()).unwrap();
    let tools = client.tools().await.unwrap();
    assert!(tools.iter().any(|t| t.function.name == "calculator"));
    let result = client
        .invoke_tool(
            "calculator",
            serde_json::json!({ "expression": "2 + 3 * 4" }),
        )
        .await
        .unwrap();
    assert_eq!(result["value"], 14);
}

#[tokio::test]
async fn agents() {
    let client = KothaGPT::with_base_url(&base_url()).unwrap();
    let agent = client
        .create_agent(AgentSpec {
            name: "rust-agent".into(),
            description: None,
            instructions: None,
            model: "kothagpt".into(),
            tools: vec![],
            temperature: None,
        })
        .await
        .unwrap();
    let run = client.run_agent(&agent.id, "হাই").await.unwrap();
    assert_eq!(run.status, "completed");
    assert!(run.output.is_some());
    client.delete_agent(&agent.id).await.unwrap();
}

#[tokio::test]
async fn agent_stream() {
    let client = KothaGPT::with_base_url(&base_url()).unwrap();
    let agent = client
        .create_agent(AgentSpec {
            name: "rust-streamer".into(),
            description: None,
            instructions: None,
            model: "kothagpt".into(),
            tools: vec![],
            temperature: None,
        })
        .await
        .unwrap();
    let mut stream = client.agent_stream(&agent.id, "হ্যালো").await.unwrap();
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event.unwrap().event);
    }
    assert!(events.contains(&"run.created".to_string()));
    assert!(events.contains(&"run.completed".to_string()));
}

#[tokio::test]
async fn websocket_chat() {
    let token = env::var("KOTHAGPT_API_TOKEN").ok();
    let mut ws = WebSocketClient::connect_with_api_key(&ws_base_url(), token.as_deref())
        .await
        .unwrap();
    let completion = ws.chat(vec![Message::user("হ্যালো")]).await.unwrap();
    assert!(!completion.text().is_empty());
}
