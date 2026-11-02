import Link from "next/link";
import {
  ArrowRight,
  Brain,
  Database,
  FlaskConical,
  MessagesSquare,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

const features = [
  {
    icon: MessagesSquare,
    title: "Chat",
    description: "Streaming chat with model selection, RAG and tool calls.",
  },
  {
    icon: Brain,
    title: "Models",
    description: "Deploy, version and compare Bangla-first foundation models.",
  },
  {
    icon: Database,
    title: "Datasets",
    description: "Curate, version and inspect your training corpora.",
  },
  {
    icon: FlaskConical,
    title: "Evaluation",
    description:
      "Benchmark models against Bangla QA, reasoning and safety suites.",
  },
];

export default function Home() {
  return (
    <main className="min-h-svh">
      <nav className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <span className="text-lg font-semibold tracking-tight">Kotha GPT</span>
        <div className="flex items-center gap-4 text-sm">
          <Link
            href="/dashboard/models"
            className="text-muted-foreground hover:text-foreground"
          >
            Models
          </Link>
          <Link
            href="/dashboard/playground"
            className="text-muted-foreground hover:text-foreground"
          >
            Playground
          </Link>
          <Button asChild variant="outline" size="sm">
            <Link href="/dashboard">Open dashboard</Link>
          </Button>
        </div>
      </nav>

      <section className="mx-auto flex max-w-6xl flex-col items-center px-6 py-24 text-center">
        <h1 className="max-w-3xl text-4xl font-bold tracking-tight sm:text-6xl">
          Bangla-first AI, in your hands.
        </h1>
        <p className="mt-6 max-w-xl text-lg text-muted-foreground">
          Kotha GPT is the platform for building, training and deploying Bengali
          and English language models — from dataset to deployment.
        </p>
        <div className="mt-8 flex gap-3">
          <Button asChild size="lg">
            <Link href="/dashboard/chat">
              Start chatting <ArrowRight />
            </Link>
          </Button>
          <Button asChild size="lg" variant="outline">
            <Link href="/dashboard/playground">Try the playground</Link>
          </Button>
        </div>
      </section>

      <section className="mx-auto max-w-6xl px-6 pb-24">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {features.map((f) => (
            <Card key={f.title} className="bg-card">
              <CardHeader>
                <f.icon className="mb-2 size-5 text-primary" />
                <CardTitle>{f.title}</CardTitle>
                <CardDescription>{f.description}</CardDescription>
              </CardHeader>
            </Card>
          ))}
        </div>
      </section>
    </main>
  );
}
