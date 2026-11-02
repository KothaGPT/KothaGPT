"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { dashboardNav } from "@/lib/config/nav";
import { buttonVariants } from "@/components/ui/button";

export function SidebarContent() {
  const pathname = usePathname();

  return (
    <nav className="flex-1 overflow-y-auto px-3 py-4">
      {dashboardNav.map((group) => (
        <div key={group.label} className="mb-6">
          <p className="px-3 pb-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
            {group.label}
          </p>
          <div className="flex flex-col gap-0.5">
            {group.items.map((item) => {
              const active =
                pathname === item.href ||
                (item.href !== "/dashboard" && pathname.startsWith(item.href));
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    buttonVariants({ variant: "ghost", size: "sm" }),
                    "justify-start gap-3 font-normal",
                    active &&
                      "bg-accent text-accent-foreground hover:bg-accent",
                  )}
                >
                  <item.icon className="size-4" />
                  {item.title}
                </Link>
              );
            })}
          </div>
        </div>
      ))}
    </nav>
  );
}
