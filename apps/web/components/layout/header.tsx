"use client";

import * as React from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Menu, Search } from "lucide-react";
import { authApi } from "@/lib/api/auth";
import { setAccessToken } from "@/lib/api/client";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  Sheet,
  SheetContent,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { SidebarContent } from "@/components/layout/sidebar-content";
import { useUIStore } from "@/stores/ui-store";

export function Header() {
  const setCommandOpen = useUIStore((s) => s.setCommandOpen);
  const router = useRouter();

  const handleSignOut = async () => {
    try {
      await authApi.logout();
    } finally {
      setAccessToken(null);
      router.push("/");
      router.refresh();
    }
  };
  return (
    <header className="sticky top-0 z-40 flex h-14 items-center gap-4 border-b border-border bg-background/95 px-4 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <Sheet>
        <SheetTrigger asChild>
          <Button variant="ghost" size="icon" className="md:hidden">
            <Menu className="size-5" />
            <span className="sr-only">Toggle navigation</span>
          </Button>
        </SheetTrigger>
        <SheetContent side="left" className="p-0">
          <SheetTitle className="sr-only">Navigation</SheetTitle>
          <SidebarContent />
        </SheetContent>
      </Sheet>

      <Link href="/dashboard" className="flex items-center gap-2">
        <span className="text-sm font-semibold tracking-tight">Kotha GPT</span>
      </Link>

      <div className="flex-1" />

      <Button
        variant="outline"
        size="sm"
        className="hidden text-muted-foreground sm:flex w-56 justify-between gap-2"
        onClick={() => setCommandOpen(true)}
      >
        <span className="flex items-center gap-2">
          <Search className="size-4" />
          Search…
        </span>
        <kbd className="pointer-events-none inline-flex h-5 select-none items-center gap-1 rounded border border-border bg-muted px-1.5 font-mono text-[10px] font-medium text-muted-foreground">
          ⌘K
        </kbd>
      </Button>

      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="icon" className="rounded-full">
            <Avatar className="size-8">
              <AvatarFallback className="bg-primary text-primary-foreground">
                AD
              </AvatarFallback>
            </Avatar>
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-56">
          <DropdownMenuLabel className="font-normal">
            <div className="flex flex-col space-y-1">
              <p className="text-sm font-medium leading-none">Abdullah</p>
              <p className="text-xs leading-none text-muted-foreground">
                admin@kothagpt.ai
              </p>
            </div>
          </DropdownMenuLabel>
          <DropdownMenuSeparator />
          <DropdownMenuItem asChild>
            <Link href="/dashboard/settings">Settings</Link>
          </DropdownMenuItem>
          <DropdownMenuItem asChild>
            <Link href="/dashboard/api-keys">API Keys</Link>
          </DropdownMenuItem>
          <DropdownMenuSeparator />
          <DropdownMenuItem
            className="text-destructive"
            onClick={handleSignOut}
          >
            Sign out
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </header>
  );
}
