import { Construction } from "lucide-react";
import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export function ComingSoon({
  title,
  description,
}: {
  title: string;
  description: string;
}) {
  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        <p className="text-sm text-muted-foreground">{description}</p>
      </div>
      <Card className="flex flex-col items-center justify-center gap-4 py-24 text-center">
        <Construction className="size-10 text-muted-foreground" />
        <CardHeader>
          <CardTitle>Coming soon</CardTitle>
          <CardDescription>
            This workspace is part of the roadmap and will be implemented in a
            later sprint.
          </CardDescription>
        </CardHeader>
      </Card>
    </div>
  );
}
