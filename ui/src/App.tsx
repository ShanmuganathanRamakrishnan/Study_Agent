import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ChatView } from '@/components/ChatView'
import { StudyPanel } from '@/components/StudyPanel'
import { UploadPanel } from '@/components/UploadPanel'
import { StatusBar } from '@/components/StatusBar'
import DarkVeil from '@/components/backgrounds/DarkVeil'
import { MessageCircle, BookOpen, Upload } from 'lucide-react'

function App() {
  return (
    <div className="dark min-h-screen flex items-start justify-center p-8 md:p-12 py-10 md:py-16 overflow-y-auto">
      {/* Animated background - React Bits DarkVeil, purely decorative */}
      <div className="fixed inset-0 -z-10 pointer-events-none">
        <DarkVeil />
      </div>

      {/* Main floating glass panel */}
      <div className="glass-panel w-full max-w-2xl flex flex-col my-auto">

        {/* ===== HEADER ===== */}
        <header className="section-spacing-lg pb-0">
          <div className="flex items-center justify-between">
            <div className="stack-xs">
              <h1 className="text-high-contrast">Lucent</h1>
              <p className="text-muted-subtle">
                Answers grounded in your study material.
              </p>
            </div>
            <StatusBar />
          </div>
          <div className="divider-glow mt-8" />
        </header>

        {/* ===== NAVIGATION TABS ===== */}
        <nav className="px-10 pt-8">
          <Tabs defaultValue="chat" className="h-full flex flex-col tabs-premium">
            <TabsList className="grid w-full grid-cols-3 h-14 p-1.5 bg-black/20 rounded-full border border-white/6">
              <TabsTrigger
                value="chat"
                className="rounded-full text-sm font-medium data-[state=active]:shadow-none flex items-center justify-center gap-2"
              >
                <MessageCircle className="icon-sm" />
                Chat
              </TabsTrigger>
              <TabsTrigger
                value="study"
                className="rounded-full text-sm font-medium data-[state=active]:shadow-none flex items-center justify-center gap-2"
              >
                <BookOpen className="icon-sm" />
                Study
              </TabsTrigger>
              <TabsTrigger
                value="upload"
                className="rounded-full text-sm font-medium data-[state=active]:shadow-none flex items-center justify-center gap-2"
              >
                <Upload className="icon-sm" />
                Upload
              </TabsTrigger>
            </TabsList>

            {/* ===== CONTENT AREA ===== */}
            <div className="mt-6 pb-2">
              <TabsContent value="chat" className="mt-0 min-h-[50vh]">
                <ChatView />
              </TabsContent>

              <TabsContent value="study" className="mt-0">
                <StudyPanel />
              </TabsContent>

              <TabsContent value="upload" className="mt-0">
                <UploadPanel />
              </TabsContent>
            </div>
          </Tabs>
        </nav>

        {/* ===== FOOTER ===== */}
        <footer className="mt-auto px-8 pb-6 pt-4">
          <div className="divider-subtle mb-5" />
          <p className="text-center text-muted-subtle text-xs">
            Lucent is an educational system designed to answer questions strictly from provided study materials.
          </p>
        </footer>
      </div>
    </div>
  )
}

export default App
