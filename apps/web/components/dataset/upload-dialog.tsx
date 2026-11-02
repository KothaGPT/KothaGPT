"use client";

import * as React from "react";
import { UploadCloud } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useUploadDataset } from "@/hooks";
import { cn } from "@/lib/utils";

export function DatasetUploadDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [name, setName] = React.useState("");
  const [file, setFile] = React.useState<File | null>(null);
  const [dragging, setDragging] = React.useState(false);
  const fileRef = React.useRef<HTMLInputElement>(null);
  const upload = useUploadDataset();

  const reset = () => {
    setName("");
    setFile(null);
  };

  const submit = async () => {
    if (!file) return;
    try {
      await upload.mutateAsync({ file, name: name || file.name });
    } catch {
      return;
    }
    reset();
    onOpenChange(false);
  };

  const handleOpenChange = (next: boolean) => {
    if (!next && upload.isPending) return;
    onOpenChange(next);
  };

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Upload dataset</DialogTitle>
          <DialogDescription>
            JSONL, JSON, CSV or parquet — the pipeline normalizes, filters,
            deduplicates and shards automatically.
          </DialogDescription>
        </DialogHeader>
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-2">
            <Label htmlFor="ds-name">Name</Label>
            <Input
              id="ds-name"
              placeholder="bangla-wiki-v2"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>
          <div
            className={cn(
              "flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-border p-8 text-center transition-colors",
              dragging && "border-primary bg-primary/5",
              file && "border-primary/50",
            )}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              const f = e.dataTransfer.files?.[0];
              if (f) setFile(f);
            }}
          >
            <UploadCloud className="size-8 text-muted-foreground" />
            {file ? (
              <p className="text-sm font-medium">{file.name}</p>
            ) : (
              <>
                <p className="text-sm">
                  Drag &amp; drop a file, or{" "}
                  <Button
                    type="button"
                    variant="link"
                    className="px-0"
                    onClick={() => fileRef.current?.click()}
                  >
                    browse
                  </Button>
                  <input
                    ref={fileRef}
                    type="file"
                    className="hidden"
                    accept=".jsonl,.json,.csv,.parquet,.txt,.gz"
                    onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                  />
                </p>
                <p className="text-xs text-muted-foreground">
                  Max 10 GB per file
                </p>
              </>
            )}
          </div>
          {upload.error && (
            <p className="text-sm text-destructive">
              {upload.error instanceof Error
                ? upload.error.message
                : "Upload failed. Please try again."}
            </p>
          )}
        </div>
        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={upload.isPending}
          >
            Cancel
          </Button>
          <Button
            onClick={() => void submit()}
            disabled={!file || upload.isPending}
          >
            {upload.isPending ? "Uploading…" : "Upload"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
