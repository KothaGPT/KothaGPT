import {
  Bot,
  Boxes,
  Brain,
  Database,
  FlaskConical,
  Gauge,
  KeyRound,
  Library,
  LayoutDashboard,
  MessagesSquare,
  Settings,
  Sparkles,
  SquareTerminal,
  TerminalSquare,
  type LucideIcon,
} from "lucide-react";

export interface NavItem {
  title: string;
  href: string;
  icon: LucideIcon;
}

export interface NavGroup {
  label: string;
  items: NavItem[];
}

export const dashboardNav: NavGroup[] = [
  {
    label: "Workspace",
    items: [
      { title: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
      { title: "Chat", href: "/dashboard/chat", icon: MessagesSquare },
      { title: "Projects", href: "/dashboard/projects", icon: Boxes },
    ],
  },
  {
    label: "Intelligence",
    items: [
      { title: "Models", href: "/dashboard/models", icon: Brain },
      { title: "Datasets", href: "/dashboard/datasets", icon: Database },
      { title: "Training", href: "/dashboard/training", icon: Gauge },
      {
        title: "Evaluations",
        href: "/dashboard/evaluations",
        icon: FlaskConical,
      },
      { title: "Knowledge", href: "/dashboard/knowledge", icon: Library },
      { title: "Agents", href: "/dashboard/agents", icon: Bot },
      { title: "Tools", href: "/dashboard/tools", icon: SquareTerminal },
    ],
  },
  {
    label: "Developer",
    items: [
      {
        title: "Playground",
        href: "/dashboard/playground",
        icon: TerminalSquare,
      },
      { title: "API Keys", href: "/dashboard/api-keys", icon: KeyRound },
      { title: "Usage", href: "/dashboard/usage", icon: Sparkles },
    ],
  },
  {
    label: "Account",
    items: [{ title: "Settings", href: "/dashboard/settings", icon: Settings }],
  },
];
