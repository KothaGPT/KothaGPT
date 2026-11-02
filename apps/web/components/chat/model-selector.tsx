"use client";

import { Check, ChevronsUpDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useModels } from "@/hooks";

export function ModelSelector({
  value,
  onChange,
}: {
  value: string;
  onChange: (model: string) => void;
}) {
  const { data } = useModels();
  const models = data?.items ?? [];
  const current = models.find((m) => m.id === value);

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm" className="justify-between gap-2">
          <span className="truncate">{current?.name ?? value}</span>
          <ChevronsUpDown className="size-3.5 opacity-50" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-56">
        <DropdownMenuLabel>Model</DropdownMenuLabel>
        <DropdownMenuSeparator />
        {models.length === 0 && (
          <DropdownMenuItem disabled>{current?.name ?? value}</DropdownMenuItem>
        )}
        {models.map((m) => (
          <DropdownMenuItem key={m.id} onSelect={() => onChange(m.id)}>
            <span className="flex-1 truncate">{m.name}</span>
            {m.id === value && <Check className="size-4" />}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
