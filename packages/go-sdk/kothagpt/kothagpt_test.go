package kothagpt

import (
	"context"
	"io"
	"os"
	"testing"
	"time"
)

var baseURL = getenv("KOTHAGPT_API_URL", "http://localhost:8000")

func getenv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func newTestClient(t *testing.T) *Client {
	t.Helper()
	return NewClient(baseURL)
}

func TestModelsList(t *testing.T) {
	client := newTestClient(t)
	models, err := client.Models.List(context.Background())
	if err != nil {
		t.Fatalf("List models: %v", err)
	}
	if len(models) == 0 {
		t.Fatal("expected at least one model")
	}
	if models[0].ID != "kothagpt" {
		t.Errorf("first model = %q, want %q", models[0].ID, "kothagpt")
	}
}

func TestChatCreate(t *testing.T) {
	client := newTestClient(t)
	resp, err := client.Chat.Create(context.Background(), ChatCompletionRequest{
		Messages: []Message{{Role: RoleUser, Content: "হ্যালো"}},
	})
	if err != nil {
		t.Fatalf("chat: %v", err)
	}
	if len(resp.Choices) == 0 {
		t.Fatal("expected at least one choice")
	}
	if resp.Text() == "" {
		t.Fatal("expected assistant content")
	}
	if resp.Usage.TotalTokens <= 0 {
		t.Errorf("expected usage.total_tokens > 0, got %d", resp.Usage.TotalTokens)
	}
}

func TestChatStream(t *testing.T) {
	client := newTestClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	stream, err := client.Chat.Stream(ctx, ChatCompletionRequest{
		Messages: []Message{{Role: RoleUser, Content: "হ্যালো"}},
	})
	if err != nil {
		t.Fatalf("open stream: %v", err)
	}
	defer stream.Stop()

	count := 0
	for {
		chunk, err := stream.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("read chunk: %v", err)
		}
		count++
		if len(chunk.Choices) == 0 {
			t.Fatal("chunk with no choices")
		}
	}
	if count == 0 {
		t.Fatal("expected at least one chunk")
	}
}

func TestEmbeddings(t *testing.T) {
	client := newTestClient(t)
	resp, err := client.Embeddings.Create(context.Background(), []string{"বাংলা", "ভাষা"})
	if err != nil {
		t.Fatalf("embeddings: %v", err)
	}
	if len(resp.Data) != 2 {
		t.Fatalf("expected 2 embeddings, got %d", len(resp.Data))
	}
	if len(resp.Data[0].Embedding) != 256 {
		t.Errorf("embedding dim = %d, want 256", len(resp.Data[0].Embedding))
	}
}

func TestRerank(t *testing.T) {
	client := newTestClient(t)
	resp, err := client.Rerank.Create(context.Background(), "বাংলা ভাষা", []string{"অন্য কিছু", "বাংলা ভাষা শেখা"})
	if err != nil {
		t.Fatalf("rerank: %v", err)
	}
	if len(resp.Results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(resp.Results))
	}
}

func TestTools(t *testing.T) {
	client := newTestClient(t)
	ctx := context.Background()
	tools, err := client.Tools.List(ctx)
	if err != nil {
		t.Fatalf("list tools: %v", err)
	}
	found := false
	for _, tool := range tools {
		if tool.Function.Name == "calculator" {
			found = true
		}
	}
	if !found {
		t.Fatal("calculator tool not listed")
	}
	result, err := client.Tools.Invoke(ctx, "calculator", map[string]any{"expression": "2 + 3 * 4"})
	if err != nil {
		t.Fatalf("invoke tool: %v", err)
	}
	value, ok := result.(map[string]any)["value"].(float64)
	if !ok || value != 14 {
		t.Errorf("calculator result = %v, want 14", result)
	}
}

func TestAgents(t *testing.T) {
	client := newTestClient(t)
	ctx := context.Background()
	agent, err := client.Agents.Create(ctx, AgentSpec{Name: "go-agent"})
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	if agent.ID == "" {
		t.Fatal("agent has no id")
	}
	run, err := client.Agents.Run(ctx, agent.ID, "হাই")
	if err != nil {
		t.Fatalf("run agent: %v", err)
	}
	if run.Status != "completed" {
		t.Errorf("run status = %q, want completed", run.Status)
	}
	if run.Output == nil {
		t.Fatal("run has no output")
	}
	if err := client.Agents.Delete(ctx, agent.ID); err != nil {
		t.Fatalf("delete agent: %v", err)
	}
}

func TestAgentStream(t *testing.T) {
	client := newTestClient(t)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	agent, err := client.Agents.Create(ctx, AgentSpec{Name: "go-streamer"})
	if err != nil {
		t.Fatalf("create agent: %v", err)
	}
	events := []string{}
	err = client.Agents.Stream(ctx, agent.ID, "হ্যালো", func(e *AgentStreamEvent) error {
		events = append(events, e.Event)
		return nil
	})
	if err != nil {
		t.Fatalf("stream agent: %v", err)
	}
	if !contains(events, "run.created") || !contains(events, "run.completed") {
		t.Fatalf("missing lifecycle events: %v", events)
	}
}

func TestWebSocket(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	ws, err := DialWebSocket(ctx, baseURL, os.Getenv("KOTHAGPT_API_TOKEN"))
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer ws.Close()
	completion, err := ws.Chat(ctx, []Message{{Role: RoleUser, Content: "হ্যালো"}})
	if err != nil {
		t.Fatalf("ws chat: %v", err)
	}
	if completion.Text() == "" {
		t.Fatal("expected assistant content over websocket")
	}
}

func contains(items []string, want string) bool {
	for _, item := range items {
		if item == want {
			return true
		}
	}
	return false
}
