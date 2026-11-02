import { beforeAll, afterAll, describe, expect, it } from "vitest";
import { KothaGPT, KothaGPTWebSocket } from "../src/index.js";

const BASE = process.env.KOTHAGPT_API_URL ?? "http://localhost:8000";

describe("KothaGPT TypeScript SDK", () => {
  let client: KothaGPT;

  beforeAll(() => {
    client = new KothaGPT({ baseURL: BASE });
  });

  it("lists models", async () => {
    const models = await client.models.list();
    expect(models.map((m) => m.id)).toContain("kothagpt");
  });

  it("creates a chat completion", async () => {
    const completion = await client.chat.create({
      messages: [{ role: "user", content: "হ্যালো" }],
    });
    expect(completion.object).toBe("chat.completion");
    expect(completion.choices[0].message.role).toBe("assistant");
    expect(completion.choices[0].message.content.length).toBeGreaterThan(0);
  });

  it("streams chat completions", async () => {
    const chunks: string[] = [];
    for await (const chunk of client.chat.stream({
      messages: [{ role: "user", content: "হ্যালো" }],
    })) {
      chunks.push(chunk.choices[0]?.delta?.content ?? "");
    }
    expect(chunks.length).toBeGreaterThan(0);
  });

  it("creates embeddings", async () => {
    const response = await client.embeddings.create(["বাংলা", "ভাষা"]);
    expect(response.data).toHaveLength(2);
    expect(response.data[0].embedding).toHaveLength(256);
  });

  it("reranks documents", async () => {
    const response = await client.rerank.create("বাংলা ভাষা", ["অন্য", "বাংলা ভাষা শেখা"]);
    expect(response.results).toHaveLength(2);
  });

  it("lists and invokes tools", async () => {
    const tools = await client.tools.list();
    expect(tools.some((t) => t.function.name === "calculator")).toBe(true);
    const result = await client.tools.invoke<{ value: number }>("calculator", { expression: "2 + 3 * 4" });
    expect(result.value).toBe(14);
  });

  it("manages agents", async () => {
    const agent = await client.agents.create({ name: "ts-agent" });
    expect(agent.id).toBeTruthy();
    const run = await client.agents.run(agent.id, "হাই");
    expect(run.status).toBe("completed");
    expect(run.output).toBeTruthy();
    await client.agents.delete(agent.id);
  });

  it("streams agent runs", async () => {
    const agent = await client.agents.create({ name: "ts-streamer" });
    const events: string[] = [];
    for await (const event of client.agents.stream(agent.id, "হ্যালো")) {
      events.push(event.event);
    }
    expect(events).toContain("run.created");
    expect(events).toContain("run.completed");
  });
});

describe("KothaGPTWebSocket", () => {
  it("chats over websocket", async () => {
    const token = process.env.KOTHAGPT_API_TOKEN;
    const ws = new KothaGPTWebSocket(BASE.replace("http", "ws"), token);
    await ws.connect();
    const completion = await ws.chat([{ role: "user", content: "হ্যালো" }]);
    expect(completion.choices[0].message.content).toBeTruthy();
    ws.close();
  });
});