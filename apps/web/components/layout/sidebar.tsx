import { SidebarContent } from "@/components/layout/sidebar-content";

export function Sidebar() {
  return (
    <aside className="hidden md:flex h-[calc(100vh-3.5rem)] w-60 shrink-0 flex-col border-r border-border bg-card">
      <SidebarContent />
    </aside>
  );
}
