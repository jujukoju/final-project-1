// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title IdentityRegistry
 * @notice Phase 6 – On-chain fingerprint identity registry with access control.
 *
 * Key design:
 *   - Only the designated oracle address (set at deploy time) may call registerIdentity.
 *   - Subject owns their identity; they can grant / revoke verifier access with an expiry.
 *   - verifyIdentity checks the registry and access-control, emits an AccessLogged event.
 *   - All sensitive data stays off-chain (IPFS); only hash(NIN) and IPFS CID stored here.
 */
contract IdentityRegistry {

    // ── State ────────────────────────────────────────────────────────────────

    address public immutable oracle;   // NIMC oracle – the only account that may register

    struct Identity {
        bytes32 hashNIN;       // keccak256 of the subject's NIN (privacy-preserving)
        string  ipfsCid;       // IPFS CID of the encrypted embedding
        bool    exists;
        uint256 registeredAt;
    }

    /// user address → Identity
    mapping(address => Identity) private _identities;

    struct AccessGrant {
        uint256 expiry;        // unix timestamp; 0 = no access
    }

    /// subject → verifier → grant
    mapping(address => mapping(address => AccessGrant)) private _access;

    // ── Events ────────────────────────────────────────────────────────────────

    event IdentityRegistered(
        address indexed subject,
        bytes32 indexed hashNIN,
        string          ipfsCid,
        uint256         timestamp
    );

    event AccessGranted(
        address indexed subject,
        address indexed verifier,
        uint256         expiry
    );

    event AccessRevoked(
        address indexed subject,
        address indexed verifier
    );

    event AccessLogged(
        address indexed subject,
        address indexed verifier,
        uint256         timestamp,
        bool            result
    );

    // ── Modifiers ─────────────────────────────────────────────────────────────

    modifier onlyOracle() {
        require(msg.sender == oracle, "IdentityRegistry: caller is not the oracle");
        _;
    }

    modifier identityExists(address subject) {
        require(_identities[subject].exists, "IdentityRegistry: identity not registered");
        _;
    }

    modifier authorisedVerifier(address subject) {
        AccessGrant memory grant = _access[subject][msg.sender];
        require(
            grant.expiry > block.timestamp,
            "IdentityRegistry: verifier not authorised or access expired"
        );
        _;
    }

    // ── Constructor ───────────────────────────────────────────────────────────

    /**
     * @param _oracle Address of the NIMC oracle account that may register identities.
     */
    constructor(address _oracle) {
        require(_oracle != address(0), "IdentityRegistry: oracle cannot be zero address");
        oracle = _oracle;
    }

    // ── Core functions ────────────────────────────────────────────────────────

    /**
     * @notice Register a new identity. Only callable by the oracle.
     * @param subject   Ethereum address of the subject being registered.
     * @param hashNIN   keccak256 hash of the subject's NIN.
     * @param ipfsCid   IPFS CID pointing to the encrypted embedding.
     */
    function registerIdentity(
        address subject,
        bytes32 hashNIN,
        string calldata ipfsCid
    ) external onlyOracle {
        require(subject != address(0), "IdentityRegistry: invalid subject address");
        require(hashNIN != bytes32(0), "IdentityRegistry: hashNIN cannot be empty");
        require(bytes(ipfsCid).length > 0, "IdentityRegistry: ipfsCid cannot be empty");
        require(!_identities[subject].exists, "IdentityRegistry: identity already registered");

        _identities[subject] = Identity({
            hashNIN:      hashNIN,
            ipfsCid:      ipfsCid,
            exists:       true,
            registeredAt: block.timestamp
        });

        emit IdentityRegistered(subject, hashNIN, ipfsCid, block.timestamp);
    }

    /**
     * @notice Check whether a subject is registered.
     * @return bool
     */
    function isRegistered(address subject) external view returns (bool) {
        return _identities[subject].exists;
    }

    /**
     * @notice Retrieve the IPFS CID for a registered subject.
     *         Only callable by an authorised verifier whose grant has not expired.
     * @param subject The address of the registered subject.
     * @return IPFS CID string.
     */
    function getIpfsCid(address subject)
        external
        view
        identityExists(subject)
        authorisedVerifier(subject)
        returns (string memory)
    {
        return _identities[subject].ipfsCid;
    }

    /**
     * @notice Verify that a subject is registered AND the calling verifier is authorised.
     *         Emits AccessLogged.
     * @param subject The address to verify.
     * @return result true if registered and access is valid.
     */
    function verifyIdentity(address subject)
        external
        identityExists(subject)
        authorisedVerifier(subject)
        returns (bool result)
    {
        result = true;
        emit AccessLogged(subject, msg.sender, block.timestamp, result);
    }

    /**
     * @notice Grant a verifier timed access to the caller's identity data.
     * @param verifier  Address of the verifier to grant.
     * @param expiry    Unix timestamp after which the grant is invalid.
     */
    function grantAccess(address verifier, uint256 expiry)
        external
        identityExists(msg.sender)
    {
        require(verifier != address(0), "IdentityRegistry: invalid verifier address");
        require(expiry > block.timestamp, "IdentityRegistry: expiry must be in the future");

        _access[msg.sender][verifier] = AccessGrant({ expiry: expiry });
        emit AccessGranted(msg.sender, verifier, expiry);
    }

    /**
     * @notice Revoke a verifier's access to the caller's identity data.
     * @param verifier Address whose access should be revoked.
     */
    function revokeAccess(address verifier) external identityExists(msg.sender) {
        require(
            _access[msg.sender][verifier].expiry != 0,
            "IdentityRegistry: no access grant to revoke"
        );
        _access[msg.sender][verifier].expiry = 0;
        emit AccessRevoked(msg.sender, verifier);
    }

    /**
     * @notice Check whether a verifier currently has valid access to a subject.
     */
    function hasAccess(address subject, address verifier) external view returns (bool) {
        return _access[subject][verifier].expiry > block.timestamp;
    }
}
