import { toast, useToast } from '@/hooks/use-toast';
import {
  Bird,
  Book,
  Bot,
  Code2,
  CornerDownLeft,
  HomeIcon,
  LifeBuoy,
  MailIcon,
  MessageSquare,
  Mic,
  Paperclip,
  Rabbit,
  Settings,
  Settings2,
  Share,
  SquareTerminal,
  SquareUser,
  Triangle,
  Turtle,
} from 'lucide-react';

import { Button } from '@/components/ui/button';
import {
  Drawer,
  DrawerContent,
  DrawerDescription,
  DrawerHeader,
  DrawerTitle,
  DrawerTrigger,
} from '@/components/ui/drawer';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
  TooltipProvider,
} from '@/components/ui/tooltip';
import Link from 'next/link';
import { Toaster } from '@/components/ui/toaster';
import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Transluce Monitor',
  description: 'Transluce Monitor',
  icons: {
    icon: '/favicon.ico',
  },
};

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <>
      <TooltipProvider>
        <div className="grid pl-[53px] h-screen w-screen">
          <aside className="inset-y fixed left-0 z-20 flex h-full flex-col border-r">
            <div className="border-b p-2">
              <a href="/dashboard">
                <Button variant="outline" size="icon" aria-label="Home">
                  <HomeIcon className="size-5" />
                </Button>
              </a>
            </div>
            <nav className="grid gap-1 p-2">
              <Tooltip>
                <TooltipTrigger asChild>
                  <Link href="/dashboard/chat">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="rounded-lg bg-muted"
                      aria-label="Observability"
                    >
                      <SquareTerminal className="size-5" />
                    </Button>
                  </Link>
                </TooltipTrigger>
                <TooltipContent side="right" sideOffset={5}>
                  Observability
                </TooltipContent>
              </Tooltip>
              {/* <Tooltip>
              <TooltipTrigger asChild>
                <Link href="/dashboard/status">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="rounded-lg"
                    aria-label="Status"
                  >
                    <Bot className="size-5" />
                  </Button>
                </Link>
              </TooltipTrigger>
              <TooltipContent side="right" sideOffset={5}>
                Status
              </TooltipContent>
            </Tooltip> */}
            </nav>
            <nav className="mt-auto grid gap-1 p-2">
              <Tooltip>
                <TooltipTrigger asChild>
                  <a href="http://eepurl.com/i1Wqoo" target="_blank">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="mt-auto rounded-lg"
                      aria-label="Help"
                    >
                      <MailIcon className="size-5" />
                    </Button>
                  </a>
                </TooltipTrigger>
                <TooltipContent side="right" sideOffset={5}>
                  Sign up for our mailing list
                </TooltipContent>
              </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <a href="mailto:info@transluce.org" target="_blank">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="mt-auto rounded-lg"
                      aria-label="Account"
                    >
                      <MessageSquare className="size-5" />
                    </Button>
                  </a>
                </TooltipTrigger>
                <TooltipContent side="right" sideOffset={5}>
                  Leave feedback
                </TooltipContent>
              </Tooltip>
            </nav>
          </aside>
          <div className="h-screen w-[calc(100vw-53px)]">{children}</div>
        </div>
      </TooltipProvider>
      <Toaster />
    </>
  );
}
