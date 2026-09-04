export type AcquireJobLike = {
  failed?: boolean
  status?: string
} | null | undefined

/** Local Investigate acquire lock. Backend remains the duplicate-job authority. */
export function investigateIsAcquiring(busy: string, job?: AcquireJobLike): boolean {
  return (
    busy === 'ACQUIRE_DUE_DILIGENCE'
    || busy === 'ACQUIRE_DUE_DILIGENCE_ALL'
    || Boolean(job && !job.failed && job.status !== 'SUCCEEDED')
  )
}
