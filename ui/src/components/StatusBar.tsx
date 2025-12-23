import { useEffect, useState } from 'react'
import { getSystemStatus, type SystemStatus } from '@/lib/api'
import { Loader2, CheckCircle2, AlertTriangle, Database } from 'lucide-react'

export function StatusBar() {
    const [status, setStatus] = useState<SystemStatus | null>(null)
    const [error, setError] = useState(false)

    useEffect(() => {
        const fetchStatus = async () => {
            const { data, error: fetchError } = await getSystemStatus()
            if (fetchError) {
                setError(true)
            } else if (data) {
                setStatus(data)
                setError(false)
            }
        }

        fetchStatus()
        // Refresh every 30 seconds
        const interval = setInterval(fetchStatus, 30000)
        return () => clearInterval(interval)
    }, [])

    // Loading state
    if (!status && !error) {
        return (
            <div className="flex items-center gap-2">
                <Loader2 className="icon-sm icon-muted animate-spin" />
                <span className="text-muted-subtle text-xs">Connecting...</span>
            </div>
        )
    }

    // Error state - backend not reachable
    if (error) {
        return (
            <div className="flex items-center gap-2">
                <span className="pill-container py-1 text-xs font-medium flex items-center gap-1.5 border bg-amber-500/10 text-amber-400 border-amber-500/20">
                    <AlertTriangle className="icon-sm" />
                    Offline
                </span>
            </div>
        )
    }

    // Ready state with explicit messaging
    const hasContent = status && status.total_chunks > 0

    return (
        <div className="flex items-center gap-2">
            {/* Status badge */}
            <span className={`pill-container py-1 text-xs font-medium flex items-center gap-1.5 border ${status?.status === 'ready' && hasContent
                    ? 'bg-green-500/10 text-green-400 border-green-500/20'
                    : status?.status === 'ready'
                        ? 'bg-blue-500/10 text-blue-400 border-blue-500/20'
                        : 'bg-white/5 text-low-contrast border-white/8'
                }`}>
                {status?.status === 'ready' && hasContent && <CheckCircle2 className="icon-sm" />}
                {status?.status === 'ready' && !hasContent && <Database className="icon-sm" />}
                {status?.status === 'ready'
                    ? hasContent ? 'Ready' : 'No Materials'
                    : status?.status
                }
            </span>

            {/* Chunk count - only show if indexed */}
            {hasContent && (
                <span className="text-muted-subtle text-xs">
                    {status.total_chunks} chunks indexed
                </span>
            )}
        </div>
    )
}
