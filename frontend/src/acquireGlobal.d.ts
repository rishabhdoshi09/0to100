export {}

declare global {
  /**
   * Compatibility binding for two legacy Investigate button guards. The backend
   * is the authoritative duplicate-acquire guard and returns the already-active
   * operation for the same symbol. Keep this false until those guards are folded
   * into InvestigatePanel's local busy state.
   */
  var acquiring: boolean
}
